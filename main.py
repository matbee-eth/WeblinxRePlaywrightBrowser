import os
import json
import asyncio
import lxml.html
import re
import weblinx.utils.format as wlf
from functools import partial
from openai import OpenAI
from typing import Callable, List
from dotenv import load_dotenv
from transformers import AutoTokenizer
from weblinx.processing.dom import clean_and_prune_tree
from modeling.dmr.processing import build_formatters, represent_element_as_dict, convert_elem_dict_to_str_legacy, format_turn_for_input
from weblinx.utils.html import filter_bboxes
from sentence_transformers.util import cos_sim
from turn import Turn
from ReplayBrowserPage import ReplayBrowserPage, launch_custom_chromium
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, dot_score
from weblinx.processing.prompt import (
    build_input_records_from_selected_turns,
    find_turns_with_instructor_chat,
    format_candidates,
    format_utterances,
    format_utterances_truncated,
    get_speaker,
    multi_attempt_format_prev_turns_truncated,
)
from weblinx.processing.truncation import (
    multi_attempt_truncate_cands_turn,
    multi_attempt_truncate_dom_tree,
)

load_dotenv()  # This loads the variables from .env

# Retrieve the env variables
model_name = os.getenv('MODEL_SLUG')
api_endpoint = os.getenv('API_ENDPOINT')
tgi_api_base = api_endpoint + '/generate'

client = OpenAI(
    base_url=api_endpoint,
    api_key="-"
)
print(f"Hitting this endpoint: {tgi_api_base} with this model: {model_name}\n")

# processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/Llama-3-8b-Web")

async def main():
    global browser, playwright, page
    browser, playwright = await launch_custom_chromium()
    context = await browser.new_context()
    _page:ReplayBrowserPage = await context.new_page()
    page = _page
    print("page", _page)

    async def onChatReceived(turn: Turn):
        print ("turn speaker:", turn.get("speaker"))
        [utterance_context, prev_text] = format_turn_for_input(page, turn, format_intent=format_intent_input, return_str=False)
        query = format_turn_for_input(page, turn, format_intent=format_intent_input)
        uids = await page.seed_html_uids()
        elements_records = format_relevant_elements_for_single_turn(turn=turn)
        docs = [r['doc'] for r in elements_records]
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        encoded = model.encode(
            [query] + docs, batch_size=4, show_progress_bar=False
        )
        query_vector, doc_vectors = encoded[0], encoded[1:]
        scores = cos_sim(query_vector, doc_vectors).cpu().squeeze().tolist()
        for i, record in enumerate(elements_records):
            record['score'] = scores[i]
        elements_records.sort(key=lambda x: x['score'], reverse=True)
        elements_records = elements_records[:10]
        selected_turns = [{"replay": page, "turn": turn, "cands_turn":elements_records} for turn in page]
        format_intent = build_formatter_for_multichoice()
        input_records = build_input_records_from_selected_turns(
            selected_turns=selected_turns,
            format_intent=format_intent,
            build_prompt_records_fn=partial(
                build_prompt_records_for_llama_truncated,
                num_utterances=len(page)+1,
                format_intent=format_intent,
                tokenizer=tokenizer,
                include_images=True,
                processor=tokenizer,
            ),
            format_prompt_records_fn=None,
        )
        insert_formatted_chat_into_records(
            records=input_records, processor=tokenizer, include_output_target=True
        )
        messages = input_records[0]["prompt"]
        messages.append({
            "role": "user", 
            "content": prev_text
        })
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            max_tokens=150,
        )
        await execute_parsed_actions(chat_completion.choices[0].message.content, turn)
        await listen_and_execute_chat_command()
    import sys

    # Function to listen for user's command line input and execute page.chat with the value
    async def listen_and_execute_chat_command():
        try:
            # Listen for user input on command line
            user_input = input("Enter your command: ")
            # Execute page.chat with the user's input
            await page.chat(user_input)
            print(f"Executed chat command: {user_input}")
        except Exception as e:
            print(f"An error occurred while executing the chat command: {e}")

    async def handle_chat(turn):
        print("turn", turn)
        await onChatReceived(turn)

    page.on("chat", handle_chat)
    try:
        print("page?", page)
        await page.chat("open up gmail")
    except ValueError as e:
        print(f"Error during chat operation: {e}")

def merge_prev_turns(prev_turns_text_list, final_user_message):
    prev_turns_merged = []

    # Merge turns from the same role
    for i, turn_text in enumerate(prev_turns_text_list):
        role = get_speaker(
            turn_text,
            instructor_name="user",
            navigator_name="assistant",
            default_name="unknown",
        )

        if i > 0 and prev_turns_merged[-1]["role"] == role:
            prev_turns_merged[-1]["content"] += " " + turn_text
        else:
            prev_turns_merged.append({"role": role, "content": turn_text})

    if len(prev_turns_merged) > 0 and prev_turns_merged[-1]["role"] == "user":
        prev_turns_merged[-1]["content"] = prev_turns_merged[-1]["content"][-1] + final_user_message
    else:
        prev_turns_merged.append({"role": "user", "content": final_user_message})

    return prev_turns_merged


def get_system_prompt_template_for_llama_mc_concise(height=None, width=None):
    viewport_size = f"Viewport size: {height}h x {width}w ;\n" if height and width else ""
    sys_prompt_template = (
        "You are an AI assistant with a deep understanding of HTML "
        "and you must predict actions based on a user request, which will be executed. "
        "Use one of the following, replacing [] with an appropriate value: "
        "change(value=[str], uid=[str]) ; "
        "click(uid=[str]) ; "
        "load(url=[str]) ; "
        'say(speaker="navigator", utterance=[str]) ; '
        "scroll(x=[int], y=[int]) ; "
        "submit(uid=[str]) ;"
        "text_input(text=[str], uid=[str]) ;\n"
        "The user's first and last {num_utterances} utterances are: "
        "{utterance_context} ;\n" +
        viewport_size +
        "Only the last {num_prev_turns} turns are provided."
    )

    return sys_prompt_template


def get_candidate_prompt_template_for_llama():
    return "Here are the top candidates for this turn: {candidate_str}\n"


def get_final_user_message():
    return "Please select the best action using the correct format, do not provide any other information or explanation."

def build_prompt_records_for_llama_truncated(
    replay,
    turn,
    format_intent,
    tokenizer,
    processor,
    include_images=False,
    cands_turn=None,
    num_utterances=5,
    num_prev_turns=5,
    system_prompt_template=None,
    candidate_prompt_template=None,
    final_user_message=None,
    include_html=True,
    format_candidates_fn=partial(
        format_candidates, max_char_len=None, use_uid_as_rank=True
    ),
    merge_prev_turns_fn=merge_prev_turns,
    format_output_dict_fn: Callable = partial(
        wlf.format_output_dictionary, function_key="intent"
    ),
    max_html_tokens=700,
    max_utterance_tokens=40 * 5,
    max_prev_turns_tokens=50 * 5,
    max_candidates_tokens=65 * 10,
    add_unused_len_to_cands=True,
    allow_iterative_reduction=False,
    parser=None,
):
    """
    Parameters
    ----------
    ...
    allow_iterative_reduction : bool
        This arg is only relevant when truncate_at_center is used behind the scene (e.g. for
        multi_attempt_format_prev_turns_truncated or multi_attempt_truncate_dom_tree). If True,
        then we will allow the iterative reduction to continue until the max_tokens is reached.
        This is useful when the tokenizer output does not necessarily decrease when we remove
        tokens from the input. For example, if we remove a token that is part of a word, but
        the updated text is retokenized to the same number of tokens, then we will continue
        to remove tokens until we reach the max_tokens limit.
    """
    if system_prompt_template is None:
        system_prompt_template = get_system_prompt_template_for_llama_mc_concise()

    if candidate_prompt_template is None:
        candidate_prompt_template = get_candidate_prompt_template_for_llama()

    if final_user_message is None:
        final_user_message = get_final_user_message()

    instructor_chat_turns = find_turns_with_instructor_chat(
        replay, turn, num_prev_turns=num_prev_turns
    )
    utterance_context = format_utterances_truncated(
        instructor_chat_turns,
        tokenizer=tokenizer,
        max_tokens=max_utterance_tokens,
        num_utterances=num_utterances,
        format_utterances_fn=format_utterances,
        allow_iterative_reduction=allow_iterative_reduction,
    )

    prev_turns_text_list = multi_attempt_format_prev_turns_truncated(
        replay=replay,
        turn=turn,
        format_intent=partial(format_intent, return_as=dict),
        tokenizer=tokenizer,
        num_prev_turns=num_prev_turns,
        turn_sep=None,  # output list
        max_tokens=max_prev_turns_tokens,
        max_attempts=5,
        format_output_dict_fn=format_output_dict_fn,
        warn_after_attempts=False,
        allow_iterative_reduction=allow_iterative_reduction,
    )

    prev_turns_merged = merge_prev_turns_fn(
        prev_turns_text_list=prev_turns_text_list, final_user_message=final_user_message
    )

    sys_prompt = system_prompt_template.format(
        num_utterances=num_utterances - 1,  # 1 less since we add the first utterance
        utterance_context=utterance_context,
        height=turn.viewport_height,
        width=turn.viewport_width,
        num_prev_turns=num_prev_turns,
    )

    if include_html and turn.html not in ["", None] and cands_turn is not None:
        dom_tree_raw = lxml.html.fromstring(turn.html, parser=parser)
        dom_tree_pruned = clean_and_prune_tree(dom_tree_raw, cands_turn=cands_turn)
        trunc = multi_attempt_truncate_dom_tree(
            dom_tree=dom_tree_pruned,
            tokenizer=tokenizer,
            max_tokens=max_html_tokens,
            warn_after_attempts=False,
            allow_iterative_reduction=allow_iterative_reduction,
        )
        html = trunc["tree_repr"]
        sys_prompt = f"```html\n{html}\n```\n" + sys_prompt
    else:
        html = ""

    if cands_turn is not None:
        if add_unused_len_to_cands:
            # Add the unused length to the candidates
            num_html_tokens = len(tokenizer.tokenize(html))
            num_utter_tokens = len(tokenizer.tokenize(utterance_context))
            num_prev_turns_tokens = len(
                tokenizer.tokenize(" ".join(prev_turns_text_list))
            )
            remain_html_tokens = max_html_tokens - num_html_tokens
            remain_utter_tokens = max_utterance_tokens - num_utter_tokens
            remain_prev_turns_tokens = max_prev_turns_tokens - num_prev_turns_tokens
            remain_tokens = (
                remain_html_tokens + remain_utter_tokens + remain_prev_turns_tokens
            )
            # Add the unused length to the max_candidates_tokens
            max_candidates_tokens += remain_tokens

        cands_turn_trunc = multi_attempt_truncate_cands_turn(
            cands_turn=cands_turn,
            tokenizer=tokenizer,
            max_tokens=max_candidates_tokens,
            format_candidates_fn=format_candidates_fn,
            warn_after_attempts=False,
            allow_iterative_reduction=allow_iterative_reduction,
        )
        cand_str = format_candidates_fn(cands_turn_trunc, max_char_len=None)
        cand_prompt = candidate_prompt_template.format(candidate_str=cand_str)
        sys_prompt += "\n" + cand_prompt

    return [{"role": "system", "content": sys_prompt}, *prev_turns_merged]


def build_formatter_for_multichoice():
    format_click = partial(wlf.format_click, formatters=(wlf.format_uid,))
    format_text_input = partial(
        wlf.format_text_input,
        formatters=(
            partial(wlf.format_arg_item, name="text", max_length=200),
            wlf.format_uid,
        ),
    )
    format_change = partial(
        wlf.format_change,
        formatters=(
            partial(wlf.format_arg_item, name="value", max_length=200),
            wlf.format_uid,
        ),
    )
    format_submit = partial(wlf.format_submit, formatters=(wlf.format_uid,))
    format_load = partial(
        wlf.format_load,
        include_transition=False,
        include_timestamp=False,
        max_length=200,
    )
    format_scroll = partial(wlf.format_scroll, include_timestamp=False)

    format_say = partial(wlf.format_say, include_timestamp=False)

    format_intent_auto = partial(
        wlf.format_intent_automatically,
        format_change=format_change,
        format_click=format_click,
        format_load=format_load,
        format_say=format_say,
        format_scroll=format_scroll,
        format_submit=format_submit,
        format_text_input=format_text_input,
    )

    return format_intent_auto

def format_relevant_elements_for_single_turn(
    turn, uid_key="data-webtasks-id"
) -> List[dict]:
    bboxes_filt = filter_bboxes(
        turn.bboxes,
        viewport_height=turn.viewport_height,
        viewport_width=turn.viewport_width,
    )
    root = lxml.html.fromstring(turn.html)
    root_tree = root.getroottree()
    elements = root.xpath(f"//*[@{uid_key}]")
    elements_filt = [p for p in elements if p.attrib[uid_key] in bboxes_filt]

    records = []

    for elem in elements_filt:
        bbox = turn.bboxes[elem.attrib[uid_key]]
        elem_dict = represent_element_as_dict(elem, bbox, root_tree)
        elem_str = convert_elem_dict_to_str_legacy(elem_dict)

        record = {
            "doc": elem_str,
            "uid": elem.attrib[uid_key],
            "turn_index": turn.index,
            "elem_dict": elem_dict,
        }
        records.append(record)
    return records

import shlex
async def execute_parsed_actions(chat_response: str, turn: Turn):
    """
    Parses the chat response and dynamically executes the actions on the global `page` object.
    
    Args:
    chat_response (str): The string response containing actions to be executed.
    """
    # Extract actions using regex to find function calls and their arguments
    actions = re.findall(r'(\w+)\(([^)]*)\)', chat_response)

    print("chat_response", actions)

    # Iterate over each action and execute it dynamically
    for action, args in actions:
        # Parse arguments into a dictionary
        args_dict = {}
        args_list = shlex.split(args)
        for arg in args_list:
            key, value = arg.split('=', 1)  # Split on the first '=' only
            args_dict[key] = value

        # Check if the action requires a 'uid' and if the turn has HTML content
        if 'uid' in args_dict and not turn.has_html():
            print(f"Action {action} requires HTML content which is not present.")
            continue

        # Dynamically call the action on the global `page` object with unpacked arguments
        if hasattr(page, action):
            print("hasattr", page, action)
            func = getattr(page, action)
            if asyncio.iscoroutinefunction(func):
                print("is async?", action, args_dict)
                if 'speaker' not in args_dict:
                    args_dict['speaker'] = "navigator"
                await func(**args_dict)
                print("completed", action, args_dict)
            else:
                print("not async?", action, args_dict)
                if 'speaker' not in args_dict:
                    args_dict['speaker'] = "navigator"
                func(**args_dict)
            if action == "load":
                return
        else:
            print(f"Action {action} is not supported on the page object.")

def __insert_empty_user_content_at_first(prompt: list):
    """
    Given a list of dictionary representing the input prompt, insert an empty user content at the first position
    after system content, only if it is not already a user content. This is done in place.
    """
    if prompt[0]["role"] != "system":
        raise ValueError(
            f"First prompt must be a system prompt. Got {prompt[0]['role']} instead."
        )

    if prompt[1]["role"] != "user":
        prompt.insert(1, {"role": "user", "content": ""})

def insert_formatted_chat_into_records(
    records,
    processor,
    include_output_target=True,
    origin_key="prompt",
    text_key="text",
):
    """
    Given a list of records, insert the formatted chat into the records. This is done in place.
    Note that we need a tokenizer's `apply_chat_template` method to be available.
    """
    for i, record in enumerate(records):
        __insert_empty_user_content_at_first(record[origin_key])

        if include_output_target:
            combined = record[origin_key] + [{"role": "assistant", "content": record["output_target"]}]
        else:
            combined = record[origin_key]
        
        try:
            text = processor.apply_chat_template(
                combined, tokenize=False, add_generation_prompt=False
            )
            records[i][text_key] = text
        except Exception as e:
            print(f"Error occurred: {e}, {combined}",)
            raise Exception(f"Error occurred while processing record: {record}. Error: {e}")

format_intent_input, _ = build_formatters()

asyncio.run(main())