import datetime as dt
from functools import cached_property
import hashlib
from pathlib import Path
from typing import List, Union

from weblinx import utils, _validate_json_backend
def format_repr(cls, *attributes, **kwargs):
    """
    Generate a __repr__ method for a class with the given attributes
    """
    attrs = [f"{attr}={getattr(cls, attr)}" for attr in attributes]
    attrs += [f"{key}={value}" for key, value in kwargs.items()]
    return f'{cls.__class__.__name__}({", ".join(attrs)})'

class Turn(dict):
    def __init__(
        self,
        turn_dict: dict,
        index: int,
        base_dir: str,
        json_backend="auto",
        encoding=None,
        demo_name: str = "unnecessary",
    ):
        """
        This class represents a turn in a demonstration, and can be used as a dictionary.
        """
        super().__init__(turn_dict)
        self.index = index
        self.demo_name = demo_name
        self.base_dir = base_dir
        self.json_backend = json_backend
        self.encoding = encoding

        _validate_json_backend(json_backend)

    def __repr__(self):
        return format_repr(self, "index", "demo_name", "base_dir")

    @property
    def args(self) -> dict:
        """
        Arguments of the turn. This is a shortcut to access the arguments
        of the action, which is nested under action -> arguments. It
        returns None if the turn is not a browser turn.
        """
        action = self.get("action", {})
        if action is None:
            return None

        return action.get("arguments")

    @property
    def type(self) -> str:
        """
        Type of the turn, either 'browser' or 'extension'
        """
        return self["type"]

    @property
    def timestamp(self) -> float:
        """
        Number of seconds since the start of the demonstration
        """
        return self["timestamp"]

    @property
    def intent(self) -> str:
        """
        If the turn is a browser turn, returns the intent of the action, otherwise returns None
        """
        action = self.get("action", {})
        if action is None:
            return None

        return action.get("intent")

    @property
    def metadata(self) -> dict:
        """
        If the turn is a browser turn, returns the metadata of the action, otherwise returns None

        Note
        ----
        The metadata is nested under action -> arguments -> metadata, so this property
        is a shortcut to access it.
        """
        args = self.args

        if args is None:
            return None

        return args.get("metadata")

    @property
    def url(self) -> str:
        """
        If the turn is a browser turn, returns the URL of the action, otherwise returns None

        Note
        ----
        The URL is nested under action -> arguments -> metadata -> url, so this property
        is a shortcut to access it.
        """
        if self.metadata is None:
            return None

        return self.metadata.get("url")

    @property
    def tab_id(self) -> int:
        """
        If the turn is a browser turn, returns the tab ID of the action, otherwise returns `None`.
        The tab ID is an integer that uniquely identifies a tab in the browser. If tab_id == -1,
        then it means there was no tab ID associated with the action (e.g. if the action was to
        delete a tab); in contrast, `tab_id` being `None` means that the turn did not have a
        action metadata.

        Note
        ----
        The tab ID is nested under action -> arguments -> metadata -> tabId, so this property
        is a shortcut to access it.
        """
        if self.metadata is None:
            return None

        return self.metadata.get("tabId")

    @property
    def viewport_height(self) -> int:
        """
        If the turn is a browser turn, returns the viewport height of the action, otherwise returns `None`.
        The viewport height is an integer that represents the height of the viewport in the browser.
        """
        if self.metadata is None:
            return None

        return self.metadata.get("viewportHeight")

    @property
    def viewport_width(self) -> int:
        """
        If the turn is a browser turn, returns the viewport width of the action, otherwise returns `None`.
        The viewport width is an integer that represents the width of the viewport in the browser.
        """
        if self.metadata is None:
            return None

        return self.metadata.get("viewportWidth")

    @property
    def mouse_x(self) -> int:
        """
        If the turn is a browser turn, returns the mouse X position of the action, otherwise returns `None`.
        The mouse X position is an integer that represents the X position of the mouse in the browser.
        """
        if self.metadata is None:
            return None

        return self.metadata.get("mouseX")

    @property
    def mouse_y(self) -> int:
        """
        If the turn is a browser turn, returns the mouse Y position of the action, otherwise returns `None`.
        The mouse Y position is an integer that represents the Y position of the mouse in the browser.
        """
        if self.metadata is None:
            return None

        return self.metadata.get("mouseY")

    @property
    def client_x(self) -> int:
        """
        If the turn is a browser turn, returns the client X position of the action, otherwise returns `None`.
        The client X is the mouse X position scaled with zoom.
        """
        if self.props is None:
            return None

        return self.props.get("clientX")

    @property
    def client_y(self) -> int:
        """
        If the turn is a browser turn, returns the client Y position of the action, otherwise returns `None`.
        The client Y is the mouse Y position scaled with zoom.
        """
        if self.props is None:
            return None

        return self.props.get("clientY")

    @property
    def zoom(self) -> float:
        """
        If the turn is a browser turn, returns the zoom level of the action, otherwise returns `None`.
        The zoom level is a float that represents how much the page is zoomed in or out.
        """

        if self.metadata is not None and "zoomLevel" in self.metadata:
            return float(self.metadata["zoomLevel"])
        else:
            return None

    @property
    def element(self) -> dict:
        """
        If the turn is a browser turn, returns the element of the action, otherwise returns None

        Example
        -------

        Here's an example of the element of a click action:

        ```json
        {
            "attributes": {
                // ...
                "data-webtasks-id": "4717de80-eda2-4319",
                "display": "block",
                "type": "submit",
                "width": "100%"
            },
            "bbox": {
                "bottom": 523.328125,
                "height": 46,
                "left": 186.5,
                "right": 588.5,
                "top": 477.328125,
                "width": 402,
                "x": 186.5,
                "y": 477.328125
            },
            "innerHTML": "Search",
            "outerHTML": "<button width=\"100%\" ...>Search</button>",
            "tagName": "BUTTON",
            "textContent": "Search",
            "xpath": "id(\"root\")/main[1]/.../button[1]"
        }
        ```


        Note
        ----
        The element is nested under action -> arguments -> element, so this property
        is a shortcut to access it.
        """
        args = self.args
        if args is None:
            return None

        return args.get("element")

    @property
    def props(self) -> dict:
        """
        If the turn is a browser turn, returns the properties of the action, otherwise returns None.

        Example
        -------

        Here's an example of the properties of a click action:
        ```json
        {
            "altKey": false,
            "button": 0,
            "buttons": 1,
            "clientX": 435,
            "clientY": 500,
            // ...
            "screenX": 435,
            "screenY": 571,
            "shiftKey": false,
            "timeStamp": 31099.899999976158,
            "x": 435,
            "y": 500
        }
        ```

        Note
        ----
        The props is nested under action -> arguments -> properties, so this property
        is a shortcut to access it.
        """
        args = self.args
        if args is None:
            return None

        return args.get("properties")

    @cached_property
    def bboxes(self) -> dict:
        return self.args.get("bboxes")

    @cached_property
    def html(self) -> str:
        """
        This uses the path returned by `self.get_html_path()` to load the HTML of the turn,
        and return it as a string, or return None if the path is invalid. It relies on the
        default parameters of `self.get_html_path()` to get the path to the HTML page.

        Example
        --------
        You can use this with BeautifulSoup to parse the HTML:

        ```
        from bs4 import BeautifulSoup
        turns = wt.Replay.from_demonstrations(demo).filter_if_html_page()
        soup = BeautifulSoup(turns[0].html, "html.parser")
        ```
        
        If you want to change the default parameters, you can call `self.get_html_path()` with the
        desired parameters and then load the HTML file yourself:

        ```
        turns = wt.Replay.from_demonstrations(demo).filter_if_html_page()
        with open(turns[0].get_html_path(subdir="pages")) as f:
            html = f.read()
        ```
        """
        if not self.has_html():
            return None

        return self["state"]["page"]

    def format_text(self, max_length=50) -> str:
        """
        This returns a string representation of the action or utterance. In the case of action
        we have a combination of the tag name, text content, and intent, with the format:
        `[tag] text -> INTENT: arg`.

        If it is a chat turn, we have a combination of the speaker and utterance, with the format:
        `[say] utterance -> SPEAKER`.

        Example
        --------

        If the action is a click on a button with the text "Click here", the output will be:
        ```
        [button] Click here -> CLICK
        ```

        If the action is to input the word "world" in a text input that already has the word "Hello":
        ```
        [input] Hello -> TEXTINPUT: world
        ```
        """
        if self.type == "chat":
            speaker = self.get("speaker").lower()
            utterance = self.get("utterance").strip()

            return f"[{speaker.lower()}] -> SAY: {utterance}"

        s = ""
        if self.element is not None:
            # else, it's a browser turn
            tag = self.element["tagName"].lower().strip()
            text = self.element["textContent"].strip()
            # Remove leading \n and trailing \n
            text = text.strip("\n")
            text = utils.shorten_text(text, max_length=max_length)

            s += f"[{tag}] {text}"

        s += f" -> {self.intent.upper()}"

        # If intent has a value, append it to the string
        if (args := self.args) is None:
            return s

        data = None

        if self.intent == "textInput":
            data = args.get("text")
        elif self.intent == "change":
            data = args.get("value")
        elif self.intent == "scroll":
            data = f'x={args["scrollX"]}, y={args["scrollY"]}'
        elif self.intent == "say":
            data = args.get("text")
        elif self.intent == "copy":
            data = utils.shorten_text(args.get("selected"), max_length=max_length)
        elif self.intent == "paste":
            data = utils.shorten_text(args.get("pasted"), max_length=max_length)
        elif self.intent in ["tabcreate", "tabremove"]:
            data = args.get("properties", {}).get("tabId")
        elif self.intent == "tabswitch":
            data = f'from={args["properties"]["tabIdOrigin"]}, to={args["properties"]["tabId"]}'
        elif self.intent == "load":
            url = args["properties"].get("url") or args.get("url", "")
            data = utils.url.shorten_url(url, width=max_length)

            if args["properties"].get("transitionType"):
                quals = " ".join(args["properties"]["transitionQualifiers"])
                data += f', transition={args["properties"]["transitionType"]}'
                data += f", qualifiers=({quals})"

        if data is not None:
            s += f": {data}"

        return s

    def validate(self) -> bool:
        """
        This checks if the following keys are present:
        - type
        - timestamp

        Additionally, it must have one of the following combinations:
        - action, state
        - speaker, utterance
        """
        required_keys = ["type", "timestamp"]
        if not all([key in self for key in required_keys]):
            return False

        if self.type == "browser":
            required_keys = ["action", "state"]

        elif self.type == "chat":
            required_keys = ["speaker", "utterance"]

        else:
            return False

        return all([key in self for key in required_keys])

    def has_screenshot(self):
        """
        Returns True if the turn has a screenshot, False otherwise
        """
        return self.get("state", {}).get("screenshot") is not None

    def has_html(self):
        """
        Returns True if the turn has an associated HTML page, False otherwise
        """
        state = self.get("state")
        if state is None:
            return False

        return state.get("page") is not None

    def has_bboxes(self, subdir: str = "bboxes", page_subdir: str = "pages"):
        """
        Checks if the turn has bounding boxes
        """
        return (
            self.get_bboxes_path(
                subdir=subdir, page_subdir=page_subdir, throw_error=False
            )
            is not None
        )

    def get_screenshot_path(
        self,
        subdir: str = "screenshots",
        return_str: bool = True,
        throw_error: bool = True,
    ) -> Union[Path, str]:
        """
        Returns the path to the screenshot of the turn, throws an error if the turn does not have a screenshot

        Parameters
        ----------
        subdir: str
            Subdirectory of the demonstration directory where the HTML pages are stored.

        return_str: bool
            If True, returns the path as a string, otherwise returns a Path object

        throw_error: bool
            If True, throws an error if the turn does not have an HTML page, otherwise returns None
        """
        if not self.has_screenshot():
            if throw_error:
                raise ValueError(f"Turn {self.index} does not have a screenshot")
            else:
                return None

        path = Path(self.base_dir, self.demo_name, subdir, self.get("state")["screenshot"])

        if return_str:
            return str(path)
        else:
            return path

    def get_html_path(
        self, subdir: str = "pages", return_str: bool = True, throw_error: bool = True
    ) -> Union[Path, str]:
        """
        Returns the path to the HTML page of the turn.

        Parameters
        ----------
        subdir: str
            Subdirectory of the demonstration directory where the HTML pages are stored.

        return_str: bool
            If True, returns the path as a string, otherwise returns a Path object

        throw_error: bool
            If True, throws an error if the turn does not have an HTML page, otherwise returns None
        """
        if not self.has_html():
            if throw_error:
                raise ValueError(f"Turn {self.index} does not have an HTML page")
            else:
                return None

        path = Path(self.base_dir, self.demo_name, subdir, self["state"]["page"])

        if return_str:
            return str(path)
        else:
            return path

    def get_bboxes_path(
        self, subdir="bboxes", page_subdir="pages", throw_error: bool = True
    ) -> List[dict]:
        """
        Returns the path to the bounding boxes file for the current turn. If the turn does not have bounding boxes
        and `throw_error` is set to True, this function will raise a ValueError. Otherwise, it will return None.

        Parameters
        ----------
        subdir : str
            The subdirectory within the demonstration directory where bounding boxes are stored. Default is "bboxes".
        page_subdir : str
            The subdirectory where page files are stored. Default is "pages". This parameter is currently not used in the function body and could be considered for removal or implementation.
        throw_error : bool
            Determines whether to throw an error if bounding boxes are not found. If False, returns None in such cases.

        Returns
        -------
        Path or None
            The path to the bounding box file as a pathlib.Path object if found, otherwise None.
        """
        if not self.has_html():
            if throw_error:
                raise ValueError(
                    f"Turn {self.index} does not have an HTML page, so it does not have bounding boxes."
                )
            else:
                return None

        html_fname = self["state"]["page"]
        index, _ = utils.get_nums_from_path(html_fname)  # different from self.index!

        path = Path(self.base_dir, self.demo_name, subdir, f"bboxes-{index}.json")

        if not path.exists():
            if throw_error:
                raise ValueError(f"Turn {self.index} does not have bounding boxes")
            else:
                return None

        return path

    def get_screenshot_status(self):
        """
        Retrieves the status of the screenshot associated with the turn. The status can be 'good', 'broken',
        or None if the status is not defined, which might be the case for turns without a screenshot.

        Returns
        -------
        Optional[str]
            A string indicating the screenshot status ('good', 'broken', or None).
        """
        return self["state"].get("screenshot_status")

    def get_xpaths_dict(
        self,
        uid_key="data-webtasks-id",
        cache_dir=None,
        allow_save=True,
        check_hash=False,
        parser="lxml",
        json_backend="auto",
        encoding=None,
    ):
        """
        Retrieves the XPaths for elements in the turn's HTML page that match a specified attribute name (uid_key).
        If a cache directory is provided, it attempts to load cached XPaths before computing new ones. Newly computed
        XPaths can be saved to the cache if `allow_save` is True. If `check_hash` is True, it validates the HTML hash
        before using cached data.

        Parameters
        ----------
        uid_key : str
            The attribute name to match elements by in the HTML document.
        cache_dir : str or Path, optional
            The directory where XPaths are cached. If None, caching is disabled.
        allow_save : bool
            Whether to save newly computed XPaths to the cache directory.
        check_hash : bool
            Whether to validate the HTML hash before using cached XPaths.
        parser : str
            The parser backend to use for HTML parsing. Currently, only 'lxml' is supported.
        json_backend : str
            The backend to use for loading and saving JSON. If 'auto', chooses the best available option.
        encoding: str
            Encoding to use when reading the file. If None, it will default to the Demonstration's encoding
            specified in the constructor, or the system's default encoding if it was not specified.
        
        Returns
        -------
        dict
            A dictionary mapping unique IDs (from `uid_key`) to their corresponding XPaths in the HTML document.
        """
        if encoding is None:
            encoding = self.encoding
        
        if parser != "lxml":
            raise ValueError(f"Invalid backend '{parser}'. Must be 'lxml'.")

        if cache_dir is not None:
            cache_dir = Path(cache_dir, self.demo_name)
            cache_path = cache_dir / f"xpaths-{self.index}.json"
            if cache_path.exists():
                result = utils.auto_read_json(cache_path, backend=json_backend, encoding=encoding)
                # If the hash is different, then the HTML has changed, so we need to
                # recompute the XPaths
                if not check_hash:
                    return result["xpaths"]

                elif (
                    check_hash
                    and result["md5"] == hashlib.md5(self.html.encode()).hexdigest()
                ):
                    return result["xpaths"]

        try:
            import lxml.html
        except ImportError:
            raise ImportError(
                "The lxml package is required to use this function. Install it with `pip install lxml`."
            )

        html = self.html
        if html is None:
            return {}

        tree = lxml.html.fromstring(html).getroottree()

        elems = tree.xpath(f"//*[@{uid_key}]")
        if len(elems) == 0:
            return {}

        xpaths = {}

        for elem in elems:
            uid = elem.attrib[uid_key]
            xpath = tree.getpath(elem)
            xpaths[uid] = xpath

        if cache_dir is not None and allow_save:
            cache_dir.mkdir(parents=True, exist_ok=True)
            save_file = {
                "xpaths": xpaths,
                "md5": hashlib.md5(html.encode()).hexdigest(),
            }
            utils.auto_save_json(save_file, cache_path, backend=json_backend)

        return xpaths