from typing import Dict, List

from bs4 import BeautifulSoup
from bs4.element import Tag
from loguru import logger

from llmeng.domain.documents import PostDocument
from llmeng.domain.exceptions import ImproperlyConfigured
from llmeng.settings import settings

from .base import BasePlaywrightCrawler


class LinkedInCrawler(BasePlaywrightCrawler[PostDocument]):
    model = PostDocument

    def __init__(self, scroll_limit: int = 5, is_deprecated: bool = True) -> None:
        super().__init__(scroll_limit)
        self._is_deprecated = is_deprecated

    async def set_extra_context_options(self) -> None:
        """Not needed in Playwright as context options are set during initialization"""
        pass

    async def login(self) -> None:
        if self._is_deprecated:
            raise DeprecationWarning(
                "As LinkedIn has updated its security measures, the login() method is no longer supported."
            )

        if not settings.LINKEDIN_USERNAME or not settings.LINKEDIN_PASSWORD:
            raise ImproperlyConfigured(
                "LinkedIn scraper requires the {LINKEDIN_USERNAME} and {LINKEDIN_PASSWORD} settings."
            )

        await self.page.goto("https://www.linkedin.com/login")
        await self.page.fill("#username", settings.LINKEDIN_USERNAME)
        await self.page.fill("#password", settings.LINKEDIN_PASSWORD)
        await self.page.click(".login__form_action_container button")
        await self.page.wait_for_load_state("networkidle")

    async def extract(self, link: str, **kwargs) -> None:
        if self._is_deprecated:
            raise DeprecationWarning(
                "As LinkedIn has updated its feed structure, the extract() method is no longer supported."
            )

        if self.model.link is not None:
            old_model = self.model.find(link=link)
            if old_model is not None:
                logger.info(f"Post already exists in the database: {link}")
                return

        logger.info(f"Starting scrapping data for profile: {link}")

        await self.login()
        soup = await self._get_page_content(link)

        _ = {
            "Name": await self._scrape_section(
                soup, "h1", class_="text-heading-xlarge"
            ),
            "About": await self._scrape_section(
                soup, "div", class_="display-flex ph5 pv3"
            ),
            "Main Page": await self._scrape_section(
                soup, "div", {"id": "main-content"}
            ),
            "Experience": await self._scrape_experience(link),
            "Education": await self._scrape_education(link),
        }

        await self.page.goto(link)
        await self.page.wait_for_load_state("networkidle")
        await self.page.click(
            ".app-aware-link.profile-creator-shared-content-view__footer-action"
        )

        # Scrolling and scraping posts
        await self.scroll_page()
        html_content = await self.get_page_source()
        soup = BeautifulSoup(html_content, "html.parser")

        post_elements = soup.find_all(
            "div",
            class_="update-components-text relative update-components-update-v2__commentary",
        )
        buttons = soup.find_all("button", class_="update-components-image__image-link")
        post_images = self._extract_image_urls(buttons)

        posts = self._extract_posts(post_elements, post_images)
        logger.info(f"Found {len(posts)} posts for profile: {link}")

        user = kwargs["user"]
        self.model.bulk_insert(
            [
                PostDocument(
                    platform="linkedin",
                    content=posts[post],
                    author_id=user.id,
                    author_full_name=user.full_name,
                )
                for post in posts
            ]
        )

        logger.info(f"Finished scrapping data for profile: {link}")

    async def _scrape_section(self, soup: BeautifulSoup, *args, **kwargs) -> str:
        """Scrape a specific section of the LinkedIn profile."""
        parent_div = soup.find(*args, **kwargs)
        return parent_div.get_text(strip=True) if parent_div else ""

    def _extract_image_urls(self, buttons: List[Tag]) -> Dict[str, str]:
        """Extracts image URLs from button elements."""
        post_images = {}
        for i, button in enumerate(buttons):
            img_tag = button.find("img")
            if not isinstance(img_tag, Tag):
                raise ValueError("img not found")
            if img_tag and "src" in img_tag.attrs:
                post_images[f"Post_{i}"] = img_tag["src"]
            else:
                logger.warning("No image found in this button")
        return post_images

    async def _get_page_content(self, url: str) -> BeautifulSoup:
        """Retrieve the page content of a given URL."""
        await self.page.goto(url)
        await self.page.wait_for_load_state("networkidle")
        content = await self.get_page_source()
        return BeautifulSoup(content, "html.parser")

    def _extract_posts(
        self, post_elements: List[Tag], post_images: Dict[str, str]
    ) -> Dict[str, Dict[str, str]]:
        """Extracts post texts and combines them with their respective images."""
        posts_data = {}
        for i, post_element in enumerate(post_elements):
            post_text = post_element.get_text(strip=True, separator="\n")
            post_data = {"text": post_text}
            if f"Post_{i}" in post_images:
                post_data["image"] = post_images[f"Post_{i}"]
            posts_data[f"Post_{i}"] = post_data
        return posts_data

    async def _scrape_experience(self, profile_url: str) -> str:
        """Scrapes the Experience section of the LinkedIn profile."""
        await self.page.goto(profile_url + "/details/experience/")
        await self.page.wait_for_load_state("networkidle")
        content = await self.get_page_source()
        soup = BeautifulSoup(content, "html.parser")
        experience_content = soup.find("section", {"id": "experience-section"})
        return experience_content.get_text(strip=True) if experience_content else ""

    async def _scrape_education(self, profile_url: str) -> str:
        """Scrapes the Education section of the LinkedIn profile."""
        await self.page.goto(profile_url + "/details/education/")
        await self.page.wait_for_load_state("networkidle")
        content = await self.get_page_source()
        soup = BeautifulSoup(content, "html.parser")
        education_content = soup.find("section", {"id": "education-section"})
        return education_content.get_text(strip=True) if education_content else ""
