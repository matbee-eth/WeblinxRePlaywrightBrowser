from typing import Any, Coroutine
from playwright.async_api import Page, async_playwright, Browser
import datetime as dt
import json
from replay import Replay
from turn import Turn
import asyncio

async def launch_custom_chromium():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    original_new_context = browser.new_context

    async def custom_new_context(*args, **kwargs):
        context = await original_new_context(*args, **kwargs)
        original_new_page = context.new_page

        async def custom_new_page(*args, **kwargs) -> Coroutine[Any, Any, ReplayBrowserPage]:
            page = await original_new_page(*args, **kwargs)
            # Wrap the page with custom logic
            page = ReplayBrowserPage({"data": []}, "./files", None, "unnecessary", page)
            page.chat("")
            return page

        context.new_page = custom_new_page
        return context

    browser.new_context = custom_new_context
    return browser, playwright  # Return both browser and playwright to manage their lifecycle externally

class EventMixin:
    def __init__(self):
        self._event_handlers = {}

    def on(self, event_name, handler):
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)

    async def emit(self, event_name, *args, **kwargs):
        print("emit?", event_name)
        handlers = self._event_handlers.get(event_name, [])
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                await handler(*args, **kwargs)
            else:
                handler(*args, **kwargs)

class ReplayBrowserPage(EventMixin, Replay):
    def __init__(self, replay_json, base_dir, encoding, demo_name, page):
        EventMixin.__init__(self)
        # Initialize the Replay component with its own parameters
        Replay.__init__(self, replay_json, base_dir, encoding=encoding, demo_name=demo_name)
        # Hold a reference to the Playwright Page object
        self._page = page
    
    async def change(self, value, uid):
        selector = f'[data-webtasks-id="{uid}"]'
        await self._page.fill(selector, value)
        arguments = await self.arguments(uid=uid)
        await self.log_action(intent="change", arguments=arguments)

    async def load(self, *args, **kwargs):
        print("LOADING:::", args, kwargs)
        await self.goto(*args, **kwargs)

    def say(self, speaker, utterance):
        self.emit("say", {"speaker": speaker, "utterance": utterance})
        # self.log_action("say", {"speaker": speaker, "utterance": utterance})

    async def scroll(self, x, y):
        await self._page.mouse.wheel(x, y)
        await self.log_action("scroll", arguments={"x": x, "y": y})

    async def submit(self, uid):
        selector = f'form[data-webtasks-id="{uid}"]'
        await self._page.locator(selector).evaluate("form => form.submit()")
        arguments = await self.arguments(uid)
        await self.log_action("submit", arguments=arguments)

    async def text_input(self, text, uid):
        selector = f'[data-webtasks-id="{uid}"]'
        await self._page.fill(selector, text)
        arguments = await self.arguments(uid)
        await self.log_action("text_input", arguments=arguments)

    async def click(self, uid, **kwargs):
            if uid == None:
                return
            print("ReplayBrowserPage:click:::", uid)
            selector = f'[data-webtasks-id="{uid}"]'
            # JavaScript to add a listener to the element that captures the event data
            try:
                await self._page.evaluate(f"""(selector) => {{
                    document.querySelector(selector)?.addEventListener('click', event => {{
                        window.lastEvent = {{
                            mouseX: event.clientX,
                            mouseY: event.clientY,
                            buttons: event.buttons,
                            altKey: event.altKey,
                            ctrlKey: event.ctrlKey,
                            shiftKey: event.shiftKey,
                            metaKey: event.metaKey,
                            timeStamp: event.timeStamp
                        }};
                    }}, {{once: true}});  // Ensure the listener is removed after capturing the event
                }}""", selector)
                # Perform the click and capture the result
                await self._page.click(selector)
            except:
                print("didnt work")

            # Retrieve the event data stored in window.lastEvent
            event_data = await self._page.evaluate("() => window.lastEvent")

            # Construct arguments object with metadata and event data
            arguments = await self.arguments(uid)
            if event_data != None:
                arguments["properties"].update(event_data)
            #     "metadata": metadata,
            #     "properties": event_data  # This now contains the event details captured
            # }
            # await self.log_action("click", arguments=arguments)

            # Log the click action with detailed arguments
    
    def get_page_metadata(self):
        return self._page.evaluate("""() => ({
            url: window.location.href,
            viewportHeight: window.innerHeight,
            viewportWidth: window.innerWidth,
            zoomLevel: document.documentElement.style.zoom || 1
        })""")
    
    async def seed_html_uids(self):
        if not self._page.url.startswith("http://") and not self._page.url.startswith("https://"):
            return
        return await self._page.evaluate("""() => {
                                    let elementToBoundingBox = {};
                                    document.querySelectorAll('*').forEach(element => {
                                            if (element === document.body) return
                                            const boundingBox = element.getBoundingClientRect();
                                            const { innerHeight, innerWidth } = window;

                                            const isInViewport = !(boundingBox.top > innerHeight || boundingBox.bottom < 0 || boundingBox.left > innerWidth || boundingBox.right < 0);
                                            const isVisible = boundingBox.width > 1 && boundingBox.height > 1;

                                            if (isInViewport && isVisible) {
                                                
                                                if (!element.hasAttribute('data-webtasks-id')) {
                                                    let myuuid = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
                                                    elementToBoundingBox[myuuid] = boundingBox;
                                                    element.setAttribute('data-webtasks-id', myuuid);
                                                } else {
                                                    elementToBoundingBox[element.getAttribute('data-webtasks-id')] = boundingBox;
                                                }
                                            }
                                    });
                                    return elementToBoundingBox;
                                }""")
    
    async def arguments(self, uid=None):
        # Include additional details like URL and viewport size
        metadata = await self.get_page_metadata()
        print(metadata)
        # Construct arguments object with metadata and event data
        bboxes = await self.seed_html_uids()
        arguments = {
            "bboxes": bboxes,
            "metadata": metadata,
            "properties": {
                "tabId": self._page.context.browser.contexts.index(self._page.context),
                "transitionQualifiers": ["from_address_bar"],
                "transitionType": "typed",
                "url": metadata['url']
            }
        }
        return arguments

    async def goto(self, speaker, url):
        print("goto:::", speaker, url)
        """
        Overridden navigate function that logs the navigation action.
        """
        # Perform the navigation
        result = await self._page.goto(url)
        await self.wait_for_load_state("networkidle")
        arguments = await self.arguments()
        
        await self.log_action("load", arguments=arguments, speaker=speaker)
        return result

    async def log_action(self, intent, arguments, speaker="navigator"):
        """
        Logs an action to the replay.
        """
        image_path = f"{self.base_dir}/screenshots/{dt.datetime.now().isoformat()}.png"
        screenshot_path = await self._page.screenshot(path=image_path)
        page_html = await self._page.content()
        turn_dict = {
            "type": "browser",
            "speaker": speaker,
            "timestamp": dt.datetime.now().timestamp(),
            "action": {
                "intent": intent,
                "arguments": arguments
            },
            "state": {
                "screenshot": image_path,
                "page": page_html,
                "screenshot_status": "good" if screenshot_path else "broken"
            },
            "properties": arguments["properties"],
            "metadata": arguments["metadata"]
        }
        self.data_dict.append(turn_dict)

    async def chat(self, utterance, speaker="instructor"):
        """
        Adds a chat Turn to the Replay with a specified speaker and utterance.
        """
        image_path = f"{self.base_dir}/screenshots/{dt.datetime.now().isoformat()}.png"
        await self._page.screenshot(path=image_path)
        bboxes = await self.seed_html_uids()
        page_html = await self._page.content()
        metadata = await self.get_page_metadata()
        arguments = {
            "bboxes": bboxes,
            "metadata": metadata,
            "properties": {
                "tabId": self._page.context.browser.contexts.index(self._page.context),
                "transitionQualifiers": ["from_address_bar"],
                "transitionType": "typed",
                "url": metadata['url']
            }
        }
        turn_dict = {
            "type": "chat",
            "timestamp": dt.datetime.now().timestamp(),
            "speaker": speaker,
            "utterance": utterance,
            "state": {
                "screenshot": image_path,
                "page": page_html,
            },
            "action": {
                "intent": "say",
                "arguments": arguments
            },
            "properties": arguments["properties"],
            "metadata": arguments["metadata"]
        }
        self.data_dict.append(turn_dict)
        turn = self[-1]
        # turn = Turn(turn_dict, index=len(self.data_dict), base_dir=self.base_dir)
        print("PAGE HTML::::", metadata['url'], turn.index, turn.has_html(), turn.intent)
        await self.emit('chat', turn)
        return turn

    # Proxy other necessary methods to the self._page object
    def __getattr__(self, item):
        return getattr(self._page, item)
    # Add overrides for other methods like fill, press, etc. as needed

