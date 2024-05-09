import datetime as dt
from functools import lru_cache
from typing import Callable, Iterator, List
from turn import Turn

def format_repr(cls, *attributes, **kwargs):
    """
    Generate a __repr__ method for a class with the given attributes
    """
    attrs = [f"{attr}={getattr(cls, attr)}" for attr in attributes]
    attrs += [f"{key}={value}" for key, value in kwargs.items()]
    return f'{cls.__class__.__name__}({", ".join(attrs)})'

class Replay:
    """
    A replay is one of the core components of a demonstration. It is a list of turns, each of
    which contains the information about the state of the browser, the action that was performed,
    and the HTML element that was interacted. It also has information about the action that was
    performed at each turn, and the timestamp of the action. If the type of turn is a 'chat' turn,
    then it will contain information about what was said in the chat.
    """

    def __init__(self, replay_json: dict, base_dir: str, encoding=None, demo_name: str = "unnecessary"):
        """
        Represents a replay of a demonstration, encapsulating a sequence of turns (actions and states) within a web session.

        Parameters
        ----------
        replay_json : dict
            The JSON object containing the replay data.
        demo_name : str
            The name of the demonstration this replay belongs to.
        base_dir : str
            The base directory where the demonstration data is stored.
        encoding : str
            The encoding to use when reading files. If None, it will default to the system's default encoding.
        """
        self.data_dict = replay_json["data"]
        self.demo_name = demo_name
        self.base_dir = str(base_dir)
        self.encoding = encoding

        created = replay_json.get("created", None)
        # TODO: check if this is the correct timestamp
        if created is not None:
            self.created = dt.datetime.fromtimestamp(created)
        else:
            self.created = None

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]

        if key < 0:
            key = len(self) + key

        elif key > len(self) - 1:
            raise IndexError(
                f"Turn index {key} out of range. The replay has {len(self)} turns, so the last index is {len(self) - 1}."
            )

        return Turn(
            self.data_dict[key],
            index=key,
            demo_name=self.demo_name,
            base_dir=self.base_dir,
            encoding=self.encoding,
        )

    def __len__(self):
        return len(self.data_dict)

    def __repr__(self):
        if self.created is not None:
            created = self.created.replace(microsecond=0)
        else:
            created = "unknown"
        return format_repr(
            self,
            "num_turns",
            created=created,
            demo_name=self.demo_name,
            base_dir=self.base_dir,
        )

    def __iter__(self) -> Iterator[Turn]:
        return iter(
            Turn(turn, index=i, demo_name=self.demo_name, base_dir=self.base_dir)
            for i, turn in enumerate(self.data_dict)
        )

    @property
    def num_turns(self):
        return len(self)

    def filter_turns(self, turn_filter: Callable[[Turn], bool]) -> List[Turn]:
        """
        Filter the turns in the replay by a custom filter function that takes as input a turn object
        and returns a list of Turn objects that satisfy the filter function.
        """

        return [turn for turn in self if turn_filter(turn)]

    def filter_by_type(self, turn_type: str) -> List[Turn]:
        """
        Filters the turns in the replay based on their type ('browser' or 'chat').

        Parameters
        ----------
        turn_type : str
            The type of turns to filter by. Must be either 'browser' or 'chat'.

        Returns
        -------
        List[Turn]
            A list of Turn objects that match the specified type.
        """
        valid_types = ["browser", "chat"]
        if turn_type not in valid_types:
            raise ValueError(
                f"Invalid turn type: {turn_type}. Please choose one of: {valid_types}"
            )

        return self.filter_turns(lambda turn: turn.type == turn_type)

    def filter_by_intent(self, intent: str) -> List[Turn]:
        """
        Filter the turns in the replay by action intent
        """
        if intent == "say":
            return self.filter_turns(lambda turn: turn.type == "chat")
        else:
            return self.filter_turns(lambda turn: turn.intent == intent)

    def filter_by_intents(self, intents: List[str], *args) -> List[Turn]:
        """
        Filter the turns in the replay by a list of action intents.

        You can either pass a list of intents as the first argument, or pass the intents as
        separate arguments. For example, the following two calls are equivalent:

        ```
        replay.filter_by_intents(['click', 'scroll'])
        replay.filter_by_intents('click', 'scroll')
        ```
        """
        if not isinstance(intents, list):
            intents = [intents]

        if len(args) > 0:
            # merge the intents with the args
            intents = list(intents) + list(args)

        # Convert to set for faster lookup
        intents = set(intents)

        if "say" in intents:
            return self.filter_turns(
                lambda turn: turn.intent in intents or turn.type == "chat"
            )
        else:
            return self.filter_turns(lambda turn: turn.intent in intents)

    def validate_turns(self) -> List[bool]:
        """
        Validate all turns in the replay. If any turn is invalid, it will return False.
        If all turns are valid, it will return True. Check `Turn.validate` for more information.
        """
        return all([turn.validate() for turn in self])

    @lru_cache()
    def filter_if_screenshot(self) -> List[Turn]:
        """
        Filter the turns in the replay by whether the turn contains a screenshot
        """
        return self.filter_turns(lambda turn: turn.has_screenshot())

    def filter_if_html_page(self) -> List[Turn]:
        """
        Filter the turns in the replay by whether the turn contains an HTML page
        """
        return self.filter_turns(lambda turn: turn.has_html())

    @lru_cache()
    def list_types(self) -> List[str]:
        """
        List all turn types in the current replay (may not be exhaustive)
        """
        return list({turn.type for turn in self})

    @lru_cache()
    def list_intents(self):
        """
        List all action intents in the current replay (may not be exhaustive)
        """
        return list(set(turn.intent for turn in self))

    @lru_cache()
    def list_screenshots(self, return_str: bool = True):
        """
        List path of all screenshots in the current replay (may not be exhaustive).
        If return_str is True, return the screenshot paths as strings instead of Path objects.

        Note
        ----
        If you want to list all screenshots available for a demonstration (even ones
        that are not in the replay), use the `list_screenshots` method of the Demonstration class.
        """
        return [
            turn.get_screenshot_path(return_str=return_str)
            for turn in self
            if turn.has_screenshot()
        ]

    @lru_cache()
    def list_html_pages(self, return_str: bool = True):
        """
        List path of all HTML pages in the current replay (may not be exhaustive)
        """
        return [
            turn.html()
            for turn in self
            if turn.has_html()
        ]

    @lru_cache()
    def list_urls(self):
        """
        List all URLs in the current replay (may not be exhaustive)
        """
        return list(set(turn.url for turn in self if turn.url is not None))

    def assign_screenshot_to_turn(self, turn, method="previous_turn"):
        """
        Sets the screenshot path of a turn. This will set the screenshot path
        under the state -> screenshot key of the turn. If the turn already has a
        screenshot path, it will be overwritten. If no possible screenshot path
        can be determined, it will return None, else it will return the screenshot
        path.

        Parameters
        ----------

        turn: Turn
            Turn object to set the screenshot path for. This will modify the turn
            in-place.

        method: str
            Method to use to set the screenshot path. If 'previous_turn', it will use
            the index of  the last turn in the demonstration to determine the screenshot
            path. At the moment, this is the only method available.
        """

        if method not in ["previous_turn"]:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: 'previous_turn'"
            )

        if method == "previous_turn":
            index = turn.index - 1

            if index >= len(self) or index < 0:
                return None

            while index >= 0:
                prev_turn = self[index]

                if prev_turn.has_screenshot():
                    if "state" not in turn or turn["state"] is None:
                        turn["state"] = {}

                    turn["state"]["screenshot"] = prev_turn["state"]["screenshot"]
                    if "screenshot_status" in prev_turn["state"]:
                        turn["state"]["screenshot_status"] = prev_turn["state"][
                            "screenshot_status"
                        ]

                    return turn["state"]["screenshot"]

                index -= 1

            return None

    def assign_html_path_to_turn(self, turn, method="previous_turn"):
        """
        Sets the HTML path of a turn. This will set the HTML path
        under the state -> page key of the turn. If the turn already has an
        HTML path, it will be overwritten. If no possible HTML path
        can be determined, it will return None, else it will return the HTML
        path.

        Parameters
        ----------

        turn: Turn
            Turn object to set the HTML path for. This will modify the turn
            in-place.

        method: str
            Method to use to set the HTML path. If 'previous_turn', it will use
            the index of  the last turn in the demonstration to determine the HTML
            path. At the moment, this is the only method available.
        """

        if method not in ["previous_turn"]:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: 'previous_turn'"
            )

        if method == "previous_turn":
            index = turn.index - 1

            if index >= len(self) or index < 0:
                return None

            while index >= 0:
                prev_turn = self[index]

                if prev_turn.has_html():
                    if "state" not in turn or turn["state"] is None:
                        turn["state"] = {}

                    turn["state"]["page"] = prev_turn["state"]["page"]
                    return turn["state"]["page"]

                index -= 1

            return None