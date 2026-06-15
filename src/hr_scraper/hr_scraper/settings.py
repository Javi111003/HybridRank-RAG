BOT_NAME = "hr_scraper"

SPIDER_MODULES = ["hr_scraper.spiders"]
NEWSPIDER_MODULE = "hr_scraper.spiders"

ADDONS = {}

ROBOTSTXT_OBEY = True

CONCURRENT_REQUESTS_PER_DOMAIN = 2
DOWNLOAD_DELAY = 1
RETRY_TIMES = 5
DOWNLOAD_TIMEOUT = 300

ITEM_PIPELINES = {
    'scrapy.pipelines.files.FilesPipeline': 1,
}

FILES_STORE = 'downloads'

USER_AGENT = "Mozilla/5.0 (compatible; GacetaScraper/1.0; +https://example.com)"

FEED_EXPORT_ENCODING = "utf-8"
