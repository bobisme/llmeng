from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
from typing import Optional, Generic, Type

from playwright.async_api import Browser, async_playwright, BrowserContext, Page
from playwright.async_api import Playwright as PlaywrightInstance

from llmeng.domain.base.nosql import T


class BaseCrawler(ABC, Generic[T]):
    model: Type[T]

    @abstractmethod
    async def extract(self, link: str, **kwargs) -> None: ...


class BasePlaywrightCrawler(BaseCrawler[T], ABC):
    def __init__(
        self,
        scroll_limit: int = 5,
        headless: bool = True,
        user_data_dir: Optional[Path] = None,
    ) -> None:
        self.scroll_limit = scroll_limit
        self.headless = headless
        self.user_data_dir = user_data_dir

        self._playwright: Optional[PlaywrightInstance] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def start(self) -> None:
        """Initialize the Playwright browser and context."""
        self._playwright = await async_playwright().start()

        context_args = {
            "viewport": {"width": 1920, "height": 1080},
            "ignore_https_errors": True,
        }

        if self.user_data_dir:
            context_args["user_data_dir"] = str(self.user_data_dir)

        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-notifications",
                "--disable-extensions",
                "--disable-background-networking",
            ],
        )

        self._context = await self._browser.new_context(**context_args)
        self._page = await self._context.new_page()

        # Set default timeout to 30 seconds
        self._page.set_default_timeout(30000)

        await self.set_extra_context_options()

    async def stop(self) -> None:
        """Clean up Playwright resources."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def set_extra_context_options(self) -> None:
        """Override to set additional context options."""
        pass

    async def login(self) -> None:
        """Override to implement login logic."""
        pass

    async def scroll_page(self) -> None:
        """Scroll through the page based on the scroll limit."""
        if not self._page:
            raise RuntimeError("Page not initialized. Call start() first.")

        current_scroll = 0
        last_height = await self._page.evaluate("document.body.scrollHeight")

        while True:
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(5)  # Wait for content to load

            new_height = await self._page.evaluate("document.body.scrollHeight")
            if new_height == last_height or (
                self.scroll_limit and current_scroll >= self.scroll_limit
            ):
                break

            last_height = new_height
            current_scroll += 1

    @property
    def page(self) -> Page:
        """Get the current page instance."""
        if not self._page:
            raise RuntimeError("Page not initialized. Call start() first.")
        return self._page

    @property
    def context(self) -> BrowserContext:
        """Get the current browser context instance."""
        if not self._context:
            raise RuntimeError("Context not initialized. Call start() first.")
        return self._context

    async def get_page_source(self):
        return await self.page.content()
