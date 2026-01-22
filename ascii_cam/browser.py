"""
ASCII Browser Renderer.

Two rendering modes:
- Visual: Pixel-based ASCII art from screenshots (true ASCII rendering)
- Semantic: DOM-based structured text with actionable elements

The visual mode captures a screenshot and converts it to ASCII art.
The semantic mode extracts DOM structure for AI agents.
"""

import io
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from PIL import Image
import numpy as np

from .converter import ASCIIConverter, CharacterSets


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
    LIST_ITEM = "list_item"
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
    full_text: str = ""  # All text content including children


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
    H = "─"
    V = "│"
    TL = "┌"
    TR = "┐"
    BL = "└"
    BR = "┘"
    DTL = "╔"
    DTR = "╗"
    DBL = "╚"
    DBR = "╝"
    DH = "═"
    DV = "║"


class SemanticRenderer:
    """
    Renders DOM structure as clean, semantic ASCII.

    Focuses on meaningful content and interactive elements,
    filtering out noisy wrapper divs common in modern SPAs.
    """

    # Semantic landmark elements worth showing
    LANDMARKS = {
        ElementType.HEADER, ElementType.NAV, ElementType.MAIN,
        ElementType.FOOTER, ElementType.SECTION, ElementType.ARTICLE,
        ElementType.ASIDE, ElementType.FORM
    }

    def __init__(self, width: int = 100, max_depth: int = 6):
        self.width = width
        self.max_depth = max_depth

    def render(self, page_info: PageInfo) -> str:
        """Render page as clean semantic ASCII."""
        lines = []

        # Header box
        lines.append(self._render_header(page_info))
        lines.append("")

        # Main content - clean extraction
        if page_info.dom_tree:
            lines.append("## Page Content")
            lines.append("")
            content = self._extract_meaningful_content(page_info.dom_tree)
            lines.extend(content)
            lines.append("")

        # Interactive elements - clean list
        if page_info.interactive_elements:
            lines.append("## Interactive Elements (use these selectors)")
            lines.append("")
            elements = self._render_interactive_list(page_info.interactive_elements)
            lines.extend(elements)

        return "\n".join(lines)

    def _render_header(self, page_info: PageInfo) -> str:
        """Render page header."""
        w = self.width - 2
        title = (page_info.title or "Untitled")[:w-2]
        url = (page_info.url or "")[:w-2]

        return "\n".join([
            BoxChars.DTL + BoxChars.DH * w + BoxChars.DTR,
            BoxChars.DV + f" {title}".ljust(w) + BoxChars.DV,
            BoxChars.DV + f" {url}".ljust(w) + BoxChars.DV,
            BoxChars.DBL + BoxChars.DH * w + BoxChars.DBR,
        ])

    def _extract_meaningful_content(self, node: DOMNode, depth: int = 0) -> List[str]:
        """Extract only meaningful content, skipping wrapper noise."""
        lines = []
        indent = "  " * min(depth, 4)

        # Skip invisible
        if not node.is_visible:
            return []

        # Check if this node is meaningful
        is_landmark = node.element_type in self.LANDMARKS
        is_heading = node.element_type == ElementType.HEADING
        is_interactive = node.is_interactive
        has_direct_text = bool(node.text.strip())
        is_list_item = node.element_type == ElementType.LIST_ITEM

        # Render meaningful nodes
        if is_landmark:
            # Show landmark with box
            name = node.tag.upper()
            lines.append(f"{indent}┌─ {name} {'─' * (20 - len(name))}┐")
            # Recurse into children
            for child in node.children:
                lines.extend(self._extract_meaningful_content(child, depth + 1))
            lines.append(f"{indent}└{'─' * 23}┘")

        elif is_heading:
            level = node.tag[1] if len(node.tag) > 1 and node.tag[1].isdigit() else "1"
            text = node.full_text or node.text
            if text.strip():
                lines.append(f"{indent}{'#' * int(level)} {self._clean_text(text)}")

        elif is_interactive:
            line = self._render_interactive_node(node, indent)
            if line:
                lines.append(line)

        elif is_list_item and (node.text.strip() or node.full_text.strip()):
            text = self._clean_text(node.full_text or node.text)
            if text:
                lines.append(f"{indent}• {text}")

        elif has_direct_text and node.element_type == ElementType.PARAGRAPH:
            text = self._clean_text(node.text)
            if text and len(text) > 10:  # Skip tiny fragments
                lines.append(f"{indent}{text}")

        elif has_direct_text and depth < 3:
            # Show meaningful text at top levels
            text = self._clean_text(node.text)
            if text and len(text) > 15:
                lines.append(f"{indent}{text}")

        # For non-landmark containers, just recurse without adding structure
        if not is_landmark and not is_interactive:
            for child in node.children:
                child_lines = self._extract_meaningful_content(child, depth)
                lines.extend(child_lines)

        return lines

    def _render_interactive_node(self, node: DOMNode, indent: str) -> str:
        """Render an interactive element inline."""
        text = self._clean_text(node.full_text or node.text)
        selector = self._short_selector(node.selector)

        if node.element_type == ElementType.BUTTON:
            label = text or "Button"
            return f"{indent}[BUTTON: {label}]  → {selector}"

        elif node.element_type == ElementType.LINK:
            href = node.attributes.get('href', '')
            label = text or href[:30] or "Link"
            return f"{indent}[LINK: {label}]  → {selector}"

        elif node.element_type == ElementType.INPUT:
            input_type = node.attributes.get('type', 'text')
            placeholder = node.attributes.get('placeholder', '')
            label = placeholder or node.attributes.get('name', '') or input_type
            return f"{indent}[INPUT ({input_type}): {label}]  → {selector}"

        elif node.element_type == ElementType.TEXTAREA:
            placeholder = node.attributes.get('placeholder', 'text area')
            return f"{indent}[TEXTAREA: {placeholder}]  → {selector}"

        elif node.element_type == ElementType.SELECT:
            return f"{indent}[SELECT: dropdown]  → {selector}"

        elif node.element_type == ElementType.CHECKBOX:
            label = text or "checkbox"
            return f"{indent}[CHECKBOX: {label}]  → {selector}"

        elif node.element_type == ElementType.RADIO:
            label = text or "radio"
            return f"{indent}[RADIO: {label}]  → {selector}"

        return ""

    def _render_interactive_list(self, elements: List[InteractiveElement]) -> List[str]:
        """Render a clean list of interactive elements."""
        lines = []

        # Group by type
        buttons = [e for e in elements if e.element_type == ElementType.BUTTON]
        links = [e for e in elements if e.element_type == ElementType.LINK]
        inputs = [e for e in elements if e.element_type in (
            ElementType.INPUT, ElementType.TEXTAREA, ElementType.SELECT,
            ElementType.CHECKBOX, ElementType.RADIO
        )]

        if buttons:
            lines.append("BUTTONS:")
            for e in buttons[:15]:  # Limit
                text = self._clean_text(e.text) or "button"
                sel = self._short_selector(e.selector)
                lines.append(f"  {e.index:2d}. [{text[:30]}]  → {sel}")
            lines.append("")

        if links:
            lines.append("LINKS:")
            for e in links[:20]:  # Limit
                text = self._clean_text(e.text)
                href = e.attributes.get('href', '')[:40]
                label = text or href or "link"
                sel = self._short_selector(e.selector)
                lines.append(f"  {e.index:2d}. {label[:40]}  → {sel}")
            lines.append("")

        if inputs:
            lines.append("INPUTS:")
            for e in inputs[:15]:  # Limit
                input_type = e.attributes.get('type', 'text')
                placeholder = e.attributes.get('placeholder', '')
                name = e.attributes.get('name', '')
                label = placeholder or name or input_type
                sel = self._short_selector(e.selector)
                lines.append(f"  {e.index:2d}. [{input_type}] {label[:30]}  → {sel}")

        return lines

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        # Normalize whitespace
        text = " ".join(text.split())
        # Truncate
        if len(text) > 80:
            text = text[:77] + "..."
        return text

    def _short_selector(self, selector: str) -> str:
        """Shorten selector for display."""
        if not selector:
            return ""
        # Prefer ID
        if selector.startswith('#'):
            return selector[:40]
        # Shorten class selectors
        if '.' in selector:
            parts = selector.split('.')
            if len(parts) > 2:
                return f"{parts[0]}.{parts[1]}"
        return selector[:40]

    def _truncate(self, text: str, max_len: int) -> str:
        if not text:
            return ""
        text = " ".join(text.split())
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."


class VisualRenderer:
    """
    Renders web pages as true ASCII art from screenshots.

    Takes a screenshot and converts it to ASCII using the ASCIIConverter.
    """

    def __init__(
        self,
        width: int = 120,
        charset: str = CharacterSets.BLOCKS,
        color: bool = True,
        invert: bool = False
    ):
        self.width = width
        self.charset = charset
        self.color = color
        self.invert = invert
        self.converter = ASCIIConverter(
            width=width,
            charset=charset,
            color=color,
            invert=invert
        )

    def render(self, screenshot: bytes, page_info: Optional[PageInfo] = None) -> str:
        """Render screenshot as ASCII art."""
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(screenshot))

        # Convert to ASCII
        ascii_art = self.converter.convert(image)

        lines = []

        # Add header if page info available
        if page_info:
            lines.append(f"╔{'═' * (self.width - 2)}╗")
            title = (page_info.title or "Untitled")[:self.width - 4]
            lines.append(f"║ {title.ljust(self.width - 4)} ║")
            url = (page_info.url or "")[:self.width - 4]
            lines.append(f"║ {url.ljust(self.width - 4)} ║")
            lines.append(f"╚{'═' * (self.width - 2)}╝")
            lines.append("")

        lines.append(ascii_art)

        return "\n".join(lines)


class BrowserRenderer:
    """
    Main browser rendering class.

    Connects to a browser via Playwright and renders pages as ASCII.
    Supports both visual (screenshot) and semantic (DOM) modes.
    """

    def __init__(
        self,
        width: int = 120,
        headless: bool = True,
        browser_type: str = "chromium",
        mode: str = "visual",
        charset: str = CharacterSets.BLOCKS,
        color: bool = True
    ):
        self.width = width
        self.headless = headless
        self.browser_type = browser_type
        self.mode = mode
        self.charset = charset
        self.color = color

        # Create renderers
        self.semantic_renderer = SemanticRenderer(width=width)
        self.visual_renderer = VisualRenderer(
            width=width,
            charset=charset,
            color=color
        )

    def _get_element_type(self, tag: str, attrs: Dict[str, str]) -> ElementType:
        """Determine element type from tag and attributes."""
        tag = tag.lower()

        type_map = {
            "button": ElementType.BUTTON,
            "a": ElementType.LINK,
            "select": ElementType.SELECT,
            "textarea": ElementType.TEXTAREA,
            "img": ElementType.IMAGE,
            "video": ElementType.VIDEO,
            "form": ElementType.FORM,
            "p": ElementType.PARAGRAPH,
            "ul": ElementType.LIST,
            "ol": ElementType.LIST,
            "li": ElementType.LIST_ITEM,
            "table": ElementType.TABLE,
            "nav": ElementType.NAV,
            "header": ElementType.HEADER,
            "footer": ElementType.FOOTER,
            "main": ElementType.MAIN,
            "section": ElementType.SECTION,
            "article": ElementType.ARTICLE,
            "aside": ElementType.ASIDE,
            "div": ElementType.DIV,
            "span": ElementType.SPAN,
        }

        if tag == "input":
            input_type = attrs.get("type", "text").lower()
            if input_type == "checkbox":
                return ElementType.CHECKBOX
            elif input_type == "radio":
                return ElementType.RADIO
            elif input_type in ("submit", "button"):
                return ElementType.BUTTON
            return ElementType.INPUT

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            return ElementType.HEADING

        return type_map.get(tag, ElementType.UNKNOWN)

    def _is_interactive(self, elem_type: ElementType) -> bool:
        """Check if element type is interactive."""
        return elem_type in (
            ElementType.BUTTON, ElementType.LINK, ElementType.INPUT,
            ElementType.SELECT, ElementType.TEXTAREA, ElementType.CHECKBOX,
            ElementType.RADIO
        )

    async def _extract_dom_playwright(self, page) -> Tuple[DOMNode, List[InteractiveElement]]:
        """Extract DOM tree using Playwright with better text extraction."""

        js_code = """
        () => {
            const SKIP_TAGS = new Set(['script', 'style', 'noscript', 'svg', 'path', 'meta', 'link', 'br', 'hr']);
            const INTERACTIVE_TAGS = new Set(['a', 'button', 'input', 'select', 'textarea']);
            const LANDMARK_TAGS = new Set(['header', 'nav', 'main', 'footer', 'section', 'article', 'aside', 'form']);

            function getDirectText(element) {
                let text = '';
                for (const child of element.childNodes) {
                    if (child.nodeType === Node.TEXT_NODE) {
                        text += child.textContent;
                    }
                }
                return text.trim().replace(/\\s+/g, ' ').substring(0, 200);
            }

            function getFullText(element) {
                return (element.textContent || '').trim().replace(/\\s+/g, ' ').substring(0, 300);
            }

            function getSelector(element) {
                if (element.id) return '#' + element.id;

                const tag = element.tagName.toLowerCase();

                // Try name attribute for inputs
                if (element.name) return tag + '[name="' + element.name + '"]';

                // Try meaningful classes
                if (element.className && typeof element.className === 'string') {
                    const classes = element.className.trim().split(/\\s+/)
                        .filter(c => c && !c.match(/^(w-|h-|p-|m-|flex|grid|text-|bg-|border)/))
                        .slice(0, 2);
                    if (classes.length > 0) {
                        return tag + '.' + classes.join('.');
                    }
                }

                // Fallback: use aria-label or role
                if (element.getAttribute('aria-label')) {
                    return tag + '[aria-label="' + element.getAttribute('aria-label').substring(0, 30) + '"]';
                }

                return tag;
            }

            function extractNode(element, depth = 0) {
                if (!element || depth > 12) return null;

                const tag = element.tagName?.toLowerCase() || '';
                if (!tag || SKIP_TAGS.has(tag)) return null;

                const rect = element.getBoundingClientRect();
                const styles = window.getComputedStyle(element);

                const isVisible = styles.display !== 'none' &&
                                  styles.visibility !== 'hidden' &&
                                  styles.opacity !== '0' &&
                                  rect.width > 0 && rect.height > 0;

                // Skip invisible elements early
                if (!isVisible && depth > 2) return null;

                const attrs = {};
                for (const attr of ['href', 'type', 'name', 'placeholder', 'value', 'aria-label', 'role']) {
                    if (element.hasAttribute(attr)) {
                        attrs[attr] = element.getAttribute(attr);
                    }
                }

                // Check for button role
                const role = element.getAttribute('role');
                const isButton = tag === 'button' || role === 'button' ||
                                 (tag === 'input' && ['submit', 'button'].includes(attrs.type));

                const node = {
                    tag: tag,
                    text: getDirectText(element),
                    fullText: getFullText(element),
                    attrs: attrs,
                    bbox: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                    selector: getSelector(element),
                    isVisible: isVisible,
                    isInteractive: INTERACTIVE_TAGS.has(tag) || isButton || role === 'link',
                    isLandmark: LANDMARK_TAGS.has(tag),
                    children: []
                };

                // Only recurse for visible elements or landmarks
                if (isVisible || node.isLandmark) {
                    for (const child of element.children) {
                        const childNode = extractNode(child, depth + 1);
                        if (childNode) {
                            node.children.push(childNode);
                        }
                    }
                }

                return node;
            }

            return extractNode(document.body);
        }
        """

        dom_data = await page.evaluate(js_code)

        interactive_elements = []
        element_index = [0]

        def convert_node(data: Dict, depth: int = 0) -> Optional[DOMNode]:
            if not data:
                return None

            bbox = None
            if data.get('bbox'):
                b = data['bbox']
                bbox = BoundingBox(b['x'], b['y'], b['width'], b['height'])

            elem_type = self._get_element_type(data['tag'], data.get('attrs', {}))

            # Override if marked as interactive
            if data.get('isInteractive') and elem_type == ElementType.DIV:
                role = data.get('attrs', {}).get('role', '')
                if role == 'button':
                    elem_type = ElementType.BUTTON
                elif role == 'link':
                    elem_type = ElementType.LINK

            is_interactive = data.get('isInteractive', False) or self._is_interactive(elem_type)

            node = DOMNode(
                tag=data['tag'],
                element_type=elem_type,
                text=data.get('text', ''),
                full_text=data.get('fullText', ''),
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
                # Get better text for the element
                text = data.get('fullText', '') or data.get('text', '')
                interactive_elements.append(InteractiveElement(
                    index=element_index[0],
                    element_type=elem_type,
                    tag=data['tag'],
                    text=text,
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

    async def render_url_async(self, url: str, full_page: bool = False) -> str:
        """Render a URL as ASCII (async)."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright required. Install: pip install playwright && playwright install"
            )

        async with async_playwright() as p:
            browser_launcher = getattr(p, self.browser_type)
            browser = await browser_launcher.launch(headless=self.headless)

            try:
                page = await browser.new_page()
                await page.set_viewport_size({"width": 1280, "height": 800})
                await page.goto(url, wait_until="networkidle", timeout=30000)

                # Wait a bit for dynamic content
                await page.wait_for_timeout(1000)

                title = await page.title()
                viewport = page.viewport_size

                page_info = PageInfo(
                    url=url,
                    title=title,
                    width=viewport["width"],
                    height=viewport["height"]
                )

                if self.mode == "visual":
                    # Take screenshot and convert to ASCII
                    screenshot = await page.screenshot(full_page=full_page)
                    return self.visual_renderer.render(screenshot, page_info)
                else:
                    # Extract DOM and render semantically
                    dom_tree, interactive_elements = await self._extract_dom_playwright(page)
                    page_info.dom_tree = dom_tree
                    page_info.interactive_elements = interactive_elements
                    return self.semantic_renderer.render(page_info)

            finally:
                await browser.close()

    def render_url(self, url: str, full_page: bool = False) -> str:
        """Render a URL as ASCII (sync)."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.render_url_async(url, full_page))
                    return future.result()
            else:
                return loop.run_until_complete(self.render_url_async(url, full_page))
        except RuntimeError:
            return asyncio.run(self.render_url_async(url, full_page))

    def render_page_sync(self, page) -> str:
        """Render an existing Playwright sync page."""
        title = page.title()
        viewport = page.viewport_size
        url = page.url

        # Simplified sync JS extraction
        js_code = """
        () => {
            const SKIP_TAGS = new Set(['script', 'style', 'noscript', 'svg', 'path', 'meta', 'link']);
            const INTERACTIVE_TAGS = new Set(['a', 'button', 'input', 'select', 'textarea']);
            const LANDMARK_TAGS = new Set(['header', 'nav', 'main', 'footer', 'section', 'article', 'aside', 'form']);

            function getDirectText(el) {
                let t = '';
                for (const c of el.childNodes) if (c.nodeType === 3) t += c.textContent;
                return t.trim().replace(/\\s+/g, ' ').substring(0, 200);
            }

            function getFullText(el) {
                return (el.textContent || '').trim().replace(/\\s+/g, ' ').substring(0, 300);
            }

            function getSelector(el) {
                if (el.id) return '#' + el.id;
                const tag = el.tagName.toLowerCase();
                if (el.name) return tag + '[name="' + el.name + '"]';
                if (el.className && typeof el.className === 'string') {
                    const cls = el.className.trim().split(/\\s+/).filter(c => c && !c.match(/^(w-|h-|p-|m-|flex)/)).slice(0,2);
                    if (cls.length) return tag + '.' + cls.join('.');
                }
                return tag;
            }

            function extract(el, d = 0) {
                if (!el || d > 12) return null;
                const tag = el.tagName?.toLowerCase() || '';
                if (!tag || SKIP_TAGS.has(tag)) return null;

                const rect = el.getBoundingClientRect();
                const st = window.getComputedStyle(el);
                const vis = st.display !== 'none' && st.visibility !== 'hidden' && rect.width > 0;

                if (!vis && d > 2) return null;

                const attrs = {};
                ['href','type','name','placeholder','value','aria-label','role'].forEach(a => {
                    if (el.hasAttribute(a)) attrs[a] = el.getAttribute(a);
                });

                const role = el.getAttribute('role');
                const isBtn = tag === 'button' || role === 'button';

                const node = {
                    tag, text: getDirectText(el), fullText: getFullText(el), attrs,
                    bbox: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
                    selector: getSelector(el), isVisible: vis,
                    isInteractive: INTERACTIVE_TAGS.has(tag) || isBtn,
                    children: []
                };

                if (vis) for (const c of el.children) { const cn = extract(c, d+1); if(cn) node.children.push(cn); }
                return node;
            }
            return extract(document.body);
        }
        """

        dom_data = page.evaluate(js_code)
        interactive_elements = []
        idx = [0]

        def convert(data, depth=0):
            if not data: return None
            bbox = BoundingBox(**data['bbox']) if data.get('bbox') else None
            elem_type = self._get_element_type(data['tag'], data.get('attrs', {}))
            is_interactive = data.get('isInteractive', False) or self._is_interactive(elem_type)

            node = DOMNode(
                tag=data['tag'], element_type=elem_type, text=data.get('text', ''),
                full_text=data.get('fullText', ''), attributes=data.get('attrs', {}),
                bbox=bbox, selector=data.get('selector', ''), depth=depth,
                is_visible=data.get('isVisible', True), is_interactive=is_interactive
            )

            if is_interactive and node.is_visible:
                idx[0] += 1
                interactive_elements.append(InteractiveElement(
                    index=idx[0], element_type=elem_type, tag=data['tag'],
                    text=data.get('fullText', '') or data.get('text', ''),
                    selector=data.get('selector', ''), xpath='', bbox=bbox,
                    attributes=data.get('attrs', {})
                ))

            for c in data.get('children', []):
                cn = convert(c, depth+1)
                if cn: node.children.append(cn)
            return node

        dom_tree = convert(dom_data)
        page_info = PageInfo(
            url=url, title=title,
            width=viewport["width"] if viewport else 1280,
            height=viewport["height"] if viewport else 800,
            dom_tree=dom_tree, interactive_elements=interactive_elements
        )
        return self.renderer.render(page_info)


class AgentBrowserContext:
    """High-level interface for AI agents to interact with web pages."""

    def __init__(self, width: int = 100, headless: bool = True):
        self.width = width
        self.headless = headless
        self.renderer = BrowserRenderer(width=width, headless=headless)
        self._history: List[str] = []

    def get_page_context(
        self,
        url: Optional[str] = None,
        page=None,
        task: Optional[str] = None,
        include_actions: bool = True
    ) -> str:
        """Get full context for an AI agent."""

        if page is not None:
            try:
                ascii_content = self.renderer.render_page_sync(page)
            except Exception:
                ascii_content = self.renderer.render_page(page)
        elif url:
            ascii_content = self.renderer.render_url(url)
            self._history.append(url)
        else:
            raise ValueError("Either url or page must be provided")

        lines = ["# Browser Agent Context", ""]

        if task:
            lines.extend(["## Task", task, ""])

        if self._history:
            lines.extend(["## History", " → ".join(self._history[-3:]), ""])

        lines.extend([ascii_content, ""])

        if include_actions:
            lines.extend([
                "## Actions",
                "Use these commands with the selectors above:",
                "  click(selector)      - Click element",
                "  type(selector, text) - Enter text",
                "  select(selector, val)- Select option",
                "  scroll(up/down)      - Scroll page",
                "  navigate(url)        - Go to URL",
                ""
            ])

        return "\n".join(lines)


def render_url(
    url: str,
    width: int = 120,
    mode: str = "visual",
    color: bool = True,
    charset: str = CharacterSets.BLOCKS
) -> str:
    """
    Render a URL as ASCII.

    Args:
        url: URL to render
        width: Output width in characters
        mode: "visual" for screenshot ASCII, "semantic" for DOM extraction
        color: Enable color output (visual mode)
        charset: Character set for visual mode

    Returns:
        ASCII representation of the page
    """
    return BrowserRenderer(
        width=width,
        mode=mode,
        color=color,
        charset=charset
    ).render_url(url)


def render_url_visual(
    url: str,
    width: int = 120,
    color: bool = True,
    charset: str = CharacterSets.BLOCKS,
    full_page: bool = False
) -> str:
    """Render a URL as visual ASCII art from screenshot."""
    return BrowserRenderer(
        width=width,
        mode="visual",
        color=color,
        charset=charset
    ).render_url(url, full_page=full_page)


def render_url_semantic(url: str, width: int = 100) -> str:
    """Render a URL as semantic ASCII with interactive elements."""
    return BrowserRenderer(width=width, mode="semantic").render_url(url)


def get_agent_context(url: str, task: Optional[str] = None, width: int = 100) -> str:
    """Get browser context for an AI agent (semantic mode)."""
    return AgentBrowserContext(width=width).get_page_context(url=url, task=task)
