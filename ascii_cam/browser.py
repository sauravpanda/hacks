"""
Semantic ASCII Browser Renderer.

Renders web pages as structured ASCII art that preserves:
- DOM hierarchy and semantic structure
- Interactive elements (buttons, links, inputs, forms)
- Approximate visual layout
- Element selectors for AI agent actions

Two rendering modes:
- Visual: Pixel-based ASCII from screenshots
- Semantic: DOM-based structured ASCII with actionable elements
"""

import re
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class ElementType(Enum):
    """Types of interactive elements."""
    BUTTON = "button"
    LINK = "link"
    INPUT = "input"
    SELECT = "select"
    TEXTAREA = "textarea"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    IMAGE = "image"
    VIDEO = "video"
    FORM = "form"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    NAV = "nav"
    HEADER = "header"
    FOOTER = "footer"
    MAIN = "main"
    SECTION = "section"
    ARTICLE = "article"
    ASIDE = "aside"
    DIV = "div"
    SPAN = "span"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Element bounding box."""
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass
class InteractiveElement:
    """An interactive element on the page."""
    index: int
    element_type: ElementType
    tag: str
    text: str
    selector: str
    xpath: str
    bbox: Optional[BoundingBox]
    attributes: Dict[str, str] = field(default_factory=dict)
    is_visible: bool = True
    is_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "type": self.element_type.value,
            "tag": self.tag,
            "text": self.text[:50] + "..." if len(self.text) > 50 else self.text,
            "selector": self.selector,
            "bbox": f"({self.bbox.x:.0f},{self.bbox.y:.0f})" if self.bbox else None,
        }


@dataclass
class DOMNode:
    """Represents a node in the DOM tree."""
    tag: str
    element_type: ElementType
    text: str
    attributes: Dict[str, str]
    bbox: Optional[BoundingBox]
    children: List["DOMNode"] = field(default_factory=list)
    selector: str = ""
    xpath: str = ""
    depth: int = 0
    is_visible: bool = True
    is_interactive: bool = False


@dataclass
class PageInfo:
    """Information about the rendered page."""
    url: str
    title: str
    width: int
    height: int
    dom_tree: Optional[DOMNode] = None
    interactive_elements: List[InteractiveElement] = field(default_factory=list)


class BoxChars:
    """Box-drawing characters for ASCII layout."""
    # Single line
    H = "─"
    V = "│"
    TL = "┌"
    TR = "┐"
    BL = "└"
    BR = "┘"
    T = "┬"
    B = "┴"
    L = "├"
    R = "┤"
    X = "┼"

    # Double line
    DH = "═"
    DV = "║"
    DTL = "╔"
    DTR = "╗"
    DBL = "╚"
    DBR = "╝"

    # Rounded
    RTL = "╭"
    RTR = "╮"
    RBL = "╰"
    RBR = "╯"


class SemanticRenderer:
    """
    Renders DOM structure as semantic ASCII.

    Creates a structured text representation that:
    - Shows page layout with box-drawing characters
    - Highlights interactive elements
    - Preserves semantic meaning
    - Lists actionable elements with selectors
    """

    # Element type icons/prefixes
    ICONS = {
        ElementType.BUTTON: "[btn]",
        ElementType.LINK: "[→]",
        ElementType.INPUT: "[___]",
        ElementType.SELECT: "[▼]",
        ElementType.TEXTAREA: "[txt]",
        ElementType.CHECKBOX: "[☐]",
        ElementType.RADIO: "(○)",
        ElementType.IMAGE: "[img]",
        ElementType.VIDEO: "[▶]",
        ElementType.FORM: "form",
        ElementType.HEADING: "#",
        ElementType.NAV: "NAV",
        ElementType.HEADER: "HEADER",
        ElementType.FOOTER: "FOOTER",
        ElementType.MAIN: "MAIN",
        ElementType.SECTION: "SECTION",
        ElementType.ARTICLE: "ARTICLE",
        ElementType.ASIDE: "ASIDE",
    }

    def __init__(
        self,
        width: int = 100,
        show_hidden: bool = False,
        max_depth: int = 10,
        show_attributes: bool = False
    ):
        """
        Initialize renderer.

        Args:
            width: Output width in characters
            show_hidden: Include hidden elements
            max_depth: Maximum DOM depth to render
            show_attributes: Show element attributes
        """
        self.width = width
        self.show_hidden = show_hidden
        self.max_depth = max_depth
        self.show_attributes = show_attributes

    def render(self, page_info: PageInfo) -> str:
        """
        Render page as semantic ASCII.

        Args:
            page_info: Page information with DOM tree

        Returns:
            Semantic ASCII representation
        """
        lines = []

        # Header
        lines.append(self._render_header(page_info))
        lines.append("")

        # Page structure
        if page_info.dom_tree:
            lines.append("## Page Structure")
            lines.append("")
            structure = self._render_dom_tree(page_info.dom_tree)
            lines.extend(structure)
            lines.append("")

        # Interactive elements
        if page_info.interactive_elements:
            lines.append("## Interactive Elements")
            lines.append("")
            elements = self._render_interactive_elements(page_info.interactive_elements)
            lines.extend(elements)

        return "\n".join(lines)

    def _render_header(self, page_info: PageInfo) -> str:
        """Render page header info."""
        box_width = self.width - 2
        title = page_info.title[:box_width - 4] if page_info.title else "Untitled"
        url = page_info.url[:box_width - 4] if page_info.url else ""

        lines = [
            BoxChars.DTL + BoxChars.DH * box_width + BoxChars.DTR,
            BoxChars.DV + f" {title}".ljust(box_width) + BoxChars.DV,
            BoxChars.DV + f" {url}".ljust(box_width) + BoxChars.DV,
            BoxChars.DV + f" Size: {page_info.width}x{page_info.height}".ljust(box_width) + BoxChars.DV,
            BoxChars.DBL + BoxChars.DH * box_width + BoxChars.DBR,
        ]
        return "\n".join(lines)

    def _render_dom_tree(self, node: DOMNode, depth: int = 0) -> List[str]:
        """Recursively render DOM tree."""
        lines = []

        if depth > self.max_depth:
            return ["  " * depth + "..."]

        if not self.show_hidden and not node.is_visible:
            return []

        # Indentation
        indent = "  " * depth
        prefix = "├─ " if depth > 0 else ""

        # Get icon/label
        icon = self.ICONS.get(node.element_type, "")
        if icon:
            icon = f"{icon} "

        # Build node line
        text = self._truncate(node.text, self.width - len(indent) - len(prefix) - len(icon) - 20)

        if node.is_interactive:
            # Highlight interactive elements
            if node.element_type == ElementType.BUTTON:
                node_str = f"[ {text or 'Button'} ]"
            elif node.element_type == ElementType.LINK:
                href = node.attributes.get('href', '')[:30]
                node_str = f"→ {text or href}"
            elif node.element_type == ElementType.INPUT:
                input_type = node.attributes.get('type', 'text')
                placeholder = node.attributes.get('placeholder', '')
                node_str = f"[{input_type}: {placeholder or '___'}]"
            elif node.element_type == ElementType.SELECT:
                node_str = f"[▼ {text or 'Select'}]"
            elif node.element_type == ElementType.CHECKBOX:
                checked = "☑" if node.attributes.get('checked') else "☐"
                node_str = f"[{checked}] {text}"
            elif node.element_type == ElementType.RADIO:
                checked = "●" if node.attributes.get('checked') else "○"
                node_str = f"({checked}) {text}"
            else:
                node_str = f"{icon}{text}"
        else:
            # Structural elements
            if node.element_type in (ElementType.HEADER, ElementType.NAV, ElementType.MAIN,
                                     ElementType.FOOTER, ElementType.SECTION, ElementType.ARTICLE):
                node_str = f"┌─ {icon}{node.tag.upper()} ─┐"
            elif node.element_type == ElementType.HEADING:
                level = node.tag[1] if len(node.tag) > 1 else "1"
                node_str = f"{'#' * int(level)} {text}"
            elif text:
                node_str = f"{icon}{text}"
            else:
                node_str = f"<{node.tag}>"

        if node_str.strip():
            # Add selector hint for interactive elements
            if node.is_interactive and node.selector:
                short_selector = node.selector[:30]
                node_str += f"  #{short_selector}" if len(node.selector) <= 30 else f"  #{short_selector}..."

            lines.append(f"{indent}{prefix}{node_str}")

        # Render children
        visible_children = [c for c in node.children if self.show_hidden or c.is_visible]
        for i, child in enumerate(visible_children):
            child_lines = self._render_dom_tree(child, depth + 1)
            lines.extend(child_lines)

        return lines

    def _render_interactive_elements(self, elements: List[InteractiveElement]) -> List[str]:
        """Render list of interactive elements."""
        lines = []

        # Group by type
        by_type: Dict[ElementType, List[InteractiveElement]] = {}
        for elem in elements:
            if elem.element_type not in by_type:
                by_type[elem.element_type] = []
            by_type[elem.element_type].append(elem)

        # Render each group
        for elem_type in [ElementType.BUTTON, ElementType.LINK, ElementType.INPUT,
                          ElementType.SELECT, ElementType.CHECKBOX, ElementType.RADIO]:
            if elem_type in by_type:
                type_name = elem_type.value.upper() + "S"
                lines.append(f"### {type_name}")

                for elem in by_type[elem_type][:20]:  # Limit to 20 per type
                    text = self._truncate(elem.text, 40)
                    pos = f"@({elem.bbox.x:.0f},{elem.bbox.y:.0f})" if elem.bbox else ""

                    icon = self.ICONS.get(elem_type, "")
                    selector_hint = elem.selector[:40]

                    line = f"  {elem.index:3d}. {icon} {text or '(no text)':<40} {selector_hint}"
                    lines.append(line)

                lines.append("")

        return lines

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text to max length."""
        if not text:
            return ""
        text = " ".join(text.split())  # Normalize whitespace
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."


class BrowserRenderer:
    """
    Main browser rendering class.

    Connects to a browser via Playwright or Selenium,
    extracts DOM structure, and renders as semantic ASCII.
    """

    def __init__(
        self,
        width: int = 100,
        headless: bool = True,
        browser_type: str = "chromium"
    ):
        """
        Initialize browser renderer.

        Args:
            width: ASCII output width
            headless: Run browser in headless mode
            browser_type: Browser to use (chromium, firefox, webkit)
        """
        self.width = width
        self.headless = headless
        self.browser_type = browser_type
        self.renderer = SemanticRenderer(width=width)

        self._playwright = None
        self._browser = None
        self._page = None

    def _get_element_type(self, tag: str, attrs: Dict[str, str]) -> ElementType:
        """Determine element type from tag and attributes."""
        tag = tag.lower()

        if tag == "button":
            return ElementType.BUTTON
        elif tag == "a":
            return ElementType.LINK
        elif tag == "input":
            input_type = attrs.get("type", "text").lower()
            if input_type == "checkbox":
                return ElementType.CHECKBOX
            elif input_type == "radio":
                return ElementType.RADIO
            elif input_type in ("submit", "button"):
                return ElementType.BUTTON
            else:
                return ElementType.INPUT
        elif tag == "select":
            return ElementType.SELECT
        elif tag == "textarea":
            return ElementType.TEXTAREA
        elif tag == "img":
            return ElementType.IMAGE
        elif tag == "video":
            return ElementType.VIDEO
        elif tag == "form":
            return ElementType.FORM
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            return ElementType.HEADING
        elif tag == "p":
            return ElementType.PARAGRAPH
        elif tag in ("ul", "ol", "li"):
            return ElementType.LIST
        elif tag == "table":
            return ElementType.TABLE
        elif tag == "nav":
            return ElementType.NAV
        elif tag == "header":
            return ElementType.HEADER
        elif tag == "footer":
            return ElementType.FOOTER
        elif tag == "main":
            return ElementType.MAIN
        elif tag == "section":
            return ElementType.SECTION
        elif tag == "article":
            return ElementType.ARTICLE
        elif tag == "aside":
            return ElementType.ASIDE
        elif tag == "div":
            return ElementType.DIV
        elif tag == "span":
            return ElementType.SPAN
        else:
            return ElementType.UNKNOWN

    def _is_interactive(self, elem_type: ElementType) -> bool:
        """Check if element type is interactive."""
        return elem_type in (
            ElementType.BUTTON, ElementType.LINK, ElementType.INPUT,
            ElementType.SELECT, ElementType.TEXTAREA, ElementType.CHECKBOX,
            ElementType.RADIO
        )

    async def _extract_dom_playwright(self, page) -> Tuple[DOMNode, List[InteractiveElement]]:
        """Extract DOM tree using Playwright."""

        # JavaScript to extract DOM structure
        js_code = """
        () => {
            function extractNode(element, depth = 0) {
                if (!element || depth > 15) return null;

                const tag = element.tagName?.toLowerCase() || '';
                if (!tag || ['script', 'style', 'noscript', 'svg', 'path'].includes(tag)) {
                    return null;
                }

                const rect = element.getBoundingClientRect();
                const styles = window.getComputedStyle(element);

                const attrs = {};
                for (const attr of element.attributes || []) {
                    attrs[attr.name] = attr.value;
                }

                // Get direct text content (not from children)
                let text = '';
                for (const child of element.childNodes) {
                    if (child.nodeType === Node.TEXT_NODE) {
                        text += child.textContent.trim() + ' ';
                    }
                }
                text = text.trim().substring(0, 200);

                // Build selector
                let selector = tag;
                if (element.id) {
                    selector = '#' + element.id;
                } else if (element.className && typeof element.className === 'string') {
                    const classes = element.className.trim().split(/\\s+/).slice(0, 2);
                    if (classes.length > 0 && classes[0]) {
                        selector = tag + '.' + classes.join('.');
                    }
                }

                const isVisible = styles.display !== 'none' &&
                                  styles.visibility !== 'hidden' &&
                                  rect.width > 0 && rect.height > 0;

                const node = {
                    tag: tag,
                    text: text,
                    attrs: attrs,
                    bbox: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height
                    },
                    selector: selector,
                    isVisible: isVisible,
                    children: []
                };

                // Process children
                for (const child of element.children) {
                    const childNode = extractNode(child, depth + 1);
                    if (childNode) {
                        node.children.push(childNode);
                    }
                }

                return node;
            }

            return extractNode(document.body);
        }
        """

        dom_data = await page.evaluate(js_code)

        # Convert to our DOM structure
        interactive_elements = []
        element_index = [0]  # Use list to allow mutation in nested function

        def convert_node(data: Dict, depth: int = 0) -> Optional[DOMNode]:
            if not data:
                return None

            bbox = None
            if data.get('bbox'):
                b = data['bbox']
                bbox = BoundingBox(b['x'], b['y'], b['width'], b['height'])

            elem_type = self._get_element_type(data['tag'], data.get('attrs', {}))
            is_interactive = self._is_interactive(elem_type)

            node = DOMNode(
                tag=data['tag'],
                element_type=elem_type,
                text=data.get('text', ''),
                attributes=data.get('attrs', {}),
                bbox=bbox,
                selector=data.get('selector', ''),
                xpath='',
                depth=depth,
                is_visible=data.get('isVisible', True),
                is_interactive=is_interactive
            )

            # Track interactive elements
            if is_interactive and node.is_visible:
                element_index[0] += 1
                interactive_elements.append(InteractiveElement(
                    index=element_index[0],
                    element_type=elem_type,
                    tag=data['tag'],
                    text=data.get('text', ''),
                    selector=data.get('selector', ''),
                    xpath='',
                    bbox=bbox,
                    attributes=data.get('attrs', {}),
                    is_visible=True,
                    is_enabled=True
                ))

            # Process children
            for child_data in data.get('children', []):
                child_node = convert_node(child_data, depth + 1)
                if child_node:
                    node.children.append(child_node)

            return node

        root = convert_node(dom_data)
        return root, interactive_elements

    async def render_url_async(self, url: str) -> str:
        """
        Render a URL as semantic ASCII (async version).

        Args:
            url: URL to render

        Returns:
            Semantic ASCII representation
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright is required for browser rendering. "
                "Install with: pip install playwright && playwright install"
            )

        async with async_playwright() as p:
            # Launch browser
            browser_launcher = getattr(p, self.browser_type)
            browser = await browser_launcher.launch(headless=self.headless)

            try:
                page = await browser.new_page()
                await page.set_viewport_size({"width": 1280, "height": 800})

                # Navigate
                await page.goto(url, wait_until="networkidle", timeout=30000)

                # Get page info
                title = await page.title()
                viewport = page.viewport_size

                # Extract DOM
                dom_tree, interactive_elements = await self._extract_dom_playwright(page)

                page_info = PageInfo(
                    url=url,
                    title=title,
                    width=viewport["width"],
                    height=viewport["height"],
                    dom_tree=dom_tree,
                    interactive_elements=interactive_elements
                )

                return self.renderer.render(page_info)

            finally:
                await browser.close()

    def render_url(self, url: str) -> str:
        """
        Render a URL as semantic ASCII (sync version).

        Args:
            url: URL to render

        Returns:
            Semantic ASCII representation
        """
        import asyncio

        # Handle event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.render_url_async(url))
                    return future.result()
            else:
                return loop.run_until_complete(self.render_url_async(url))
        except RuntimeError:
            return asyncio.run(self.render_url_async(url))

    def render_page(self, page) -> str:
        """
        Render an existing Playwright page as semantic ASCII.

        Args:
            page: Playwright Page object

        Returns:
            Semantic ASCII representation
        """
        import asyncio

        async def _render():
            title = await page.title()
            viewport = page.viewport_size
            url = page.url

            dom_tree, interactive_elements = await self._extract_dom_playwright(page)

            page_info = PageInfo(
                url=url,
                title=title,
                width=viewport["width"] if viewport else 1280,
                height=viewport["height"] if viewport else 800,
                dom_tree=dom_tree,
                interactive_elements=interactive_elements
            )

            return self.renderer.render(page_info)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Return a coroutine for the caller to await
                return asyncio.ensure_future(_render())
            return loop.run_until_complete(_render())
        except RuntimeError:
            return asyncio.run(_render())

    def render_page_sync(self, page) -> str:
        """
        Render a Playwright sync page as semantic ASCII.

        Args:
            page: Playwright sync Page object

        Returns:
            Semantic ASCII representation
        """
        # For sync Playwright API
        title = page.title()
        viewport = page.viewport_size
        url = page.url

        # Use sync evaluation
        js_code = """
        () => {
            function extractNode(element, depth = 0) {
                if (!element || depth > 15) return null;
                const tag = element.tagName?.toLowerCase() || '';
                if (!tag || ['script', 'style', 'noscript', 'svg', 'path'].includes(tag)) {
                    return null;
                }
                const rect = element.getBoundingClientRect();
                const styles = window.getComputedStyle(element);
                const attrs = {};
                for (const attr of element.attributes || []) {
                    attrs[attr.name] = attr.value;
                }
                let text = '';
                for (const child of element.childNodes) {
                    if (child.nodeType === Node.TEXT_NODE) {
                        text += child.textContent.trim() + ' ';
                    }
                }
                text = text.trim().substring(0, 200);
                let selector = tag;
                if (element.id) {
                    selector = '#' + element.id;
                } else if (element.className && typeof element.className === 'string') {
                    const classes = element.className.trim().split(/\\s+/).slice(0, 2);
                    if (classes.length > 0 && classes[0]) {
                        selector = tag + '.' + classes.join('.');
                    }
                }
                const isVisible = styles.display !== 'none' &&
                                  styles.visibility !== 'hidden' &&
                                  rect.width > 0 && rect.height > 0;
                const node = {
                    tag: tag,
                    text: text,
                    attrs: attrs,
                    bbox: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                    selector: selector,
                    isVisible: isVisible,
                    children: []
                };
                for (const child of element.children) {
                    const childNode = extractNode(child, depth + 1);
                    if (childNode) node.children.push(childNode);
                }
                return node;
            }
            return extractNode(document.body);
        }
        """

        dom_data = page.evaluate(js_code)

        # Convert DOM data
        interactive_elements = []
        element_index = [0]

        def convert_node(data, depth=0):
            if not data:
                return None

            bbox = None
            if data.get('bbox'):
                b = data['bbox']
                bbox = BoundingBox(b['x'], b['y'], b['width'], b['height'])

            elem_type = self._get_element_type(data['tag'], data.get('attrs', {}))
            is_interactive = self._is_interactive(elem_type)

            node = DOMNode(
                tag=data['tag'],
                element_type=elem_type,
                text=data.get('text', ''),
                attributes=data.get('attrs', {}),
                bbox=bbox,
                selector=data.get('selector', ''),
                xpath='',
                depth=depth,
                is_visible=data.get('isVisible', True),
                is_interactive=is_interactive
            )

            if is_interactive and node.is_visible:
                element_index[0] += 1
                interactive_elements.append(InteractiveElement(
                    index=element_index[0],
                    element_type=elem_type,
                    tag=data['tag'],
                    text=data.get('text', ''),
                    selector=data.get('selector', ''),
                    xpath='',
                    bbox=bbox,
                    attributes=data.get('attrs', {}),
                    is_visible=True,
                    is_enabled=True
                ))

            for child_data in data.get('children', []):
                child_node = convert_node(child_data, depth + 1)
                if child_node:
                    node.children.append(child_node)

            return node

        dom_tree = convert_node(dom_data)

        page_info = PageInfo(
            url=url,
            title=title,
            width=viewport["width"] if viewport else 1280,
            height=viewport["height"] if viewport else 800,
            dom_tree=dom_tree,
            interactive_elements=interactive_elements
        )

        return self.renderer.render(page_info)


class AgentBrowserContext:
    """
    High-level interface for AI agents to interact with web pages.

    Provides:
    - Semantic ASCII rendering of pages
    - Interactive element listing with selectors
    - Action suggestions
    - State tracking
    """

    def __init__(self, width: int = 100, headless: bool = True):
        """
        Initialize agent browser context.

        Args:
            width: ASCII output width
            headless: Run browser in headless mode
        """
        self.width = width
        self.headless = headless
        self.renderer = BrowserRenderer(width=width, headless=headless)
        self._history: List[str] = []
        self._current_url: Optional[str] = None

    def get_page_context(
        self,
        url: Optional[str] = None,
        page=None,
        task: Optional[str] = None,
        include_actions: bool = True
    ) -> str:
        """
        Get full context for an AI agent.

        Args:
            url: URL to render (if no page provided)
            page: Existing Playwright page object
            task: Current task description
            include_actions: Include suggested actions

        Returns:
            Complete context string for LLM
        """
        # Render page
        if page is not None:
            if hasattr(page, 'title'):
                # Check if it's sync or async
                try:
                    # Try sync first
                    ascii_content = self.renderer.render_page_sync(page)
                except Exception:
                    ascii_content = self.renderer.render_page(page)
        elif url:
            ascii_content = self.renderer.render_url(url)
            self._current_url = url
            self._history.append(url)
        else:
            raise ValueError("Either url or page must be provided")

        # Build context
        lines = [
            "# Browser Agent Context",
            ""
        ]

        if task:
            lines.extend([
                "## Current Task",
                task,
                ""
            ])

        if self._history:
            lines.extend([
                "## Navigation History",
                " → ".join(self._history[-5:]),  # Last 5 pages
                ""
            ])

        lines.extend([
            "## Page Content",
            "",
            ascii_content,
            ""
        ])

        if include_actions:
            lines.extend([
                "## Available Actions",
                "",
                "You can perform these actions:",
                "- `click(selector)` - Click an element",
                "- `type(selector, text)` - Type text into an input",
                "- `select(selector, value)` - Select dropdown option",
                "- `scroll(direction)` - Scroll up/down",
                "- `navigate(url)` - Go to a URL",
                "- `back()` - Go back",
                "- `wait(ms)` - Wait for content to load",
                "",
                "Use the element selectors from the Interactive Elements list above.",
                ""
            ])

        return "\n".join(lines)

    def format_action_result(
        self,
        action: str,
        success: bool,
        message: str = "",
        new_page_content: Optional[str] = None
    ) -> str:
        """
        Format the result of an action for the agent.

        Args:
            action: The action that was performed
            success: Whether the action succeeded
            message: Additional message
            new_page_content: New page ASCII if page changed

        Returns:
            Formatted result string
        """
        status = "SUCCESS" if success else "FAILED"
        lines = [
            f"## Action Result: {status}",
            f"Action: {action}",
        ]

        if message:
            lines.append(f"Message: {message}")

        if new_page_content:
            lines.extend([
                "",
                "## Updated Page Content",
                "",
                new_page_content
            ])

        return "\n".join(lines)


def render_url(url: str, width: int = 100) -> str:
    """
    Convenience function to render a URL as semantic ASCII.

    Args:
        url: URL to render
        width: Output width

    Returns:
        Semantic ASCII representation
    """
    renderer = BrowserRenderer(width=width)
    return renderer.render_url(url)


def get_agent_context(
    url: str,
    task: Optional[str] = None,
    width: int = 100
) -> str:
    """
    Get browser context for an AI agent.

    Args:
        url: URL to render
        task: Task description
        width: Output width

    Returns:
        Agent context string
    """
    agent = AgentBrowserContext(width=width)
    return agent.get_page_context(url=url, task=task)
