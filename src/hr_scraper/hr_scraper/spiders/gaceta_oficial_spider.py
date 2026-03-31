import scrapy
from scrapy import FormRequest
from ..items import GacetaOficialItem


class GacetaOficialSpider(scrapy.Spider):
    name = "gaceta_oficial"
    allowed_domains = ["gacetaoficial.gob.cu"]
    bootstrap_url = "https://www.gacetaoficial.gob.cu/es/gacetas-oficiales"
    search_url = "https://www.gacetaoficial.gob.cu/es/getdatagaceta"

    custom_settings = {
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "DOWNLOAD_DELAY": 2,
        "RETRY_TIMES": 3,
    }

    base_request_headers = {
        "Accept": "*/*",
        "Accept-Language": "es,es-ES;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,es-MX;q=0.5",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://www.gacetaoficial.gob.cu",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0"
        ),
        "X-Requested-With": "XMLHttpRequest",
    }

    request_cookies = {"has_js": "1"}

    TOTAL_PAGES = 100

    def __init__(self, total_pages=None, start_page=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_pages = max(1, int(total_pages or self.TOTAL_PAGES))
        self.start_page = max(0, int(start_page))

    async def start(self):
        yield scrapy.Request(
            url=self.bootstrap_url,
            callback=self.bootstrap_search,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": self.base_request_headers["Accept-Language"],
                "User-Agent": self.base_request_headers["User-Agent"],
            },
            cookies=self.request_cookies,
        )

    def start_requests(self):
        yield scrapy.Request(
            url=self.bootstrap_url,
            callback=self.bootstrap_search,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": self.base_request_headers["Accept-Language"],
                "User-Agent": self.base_request_headers["User-Agent"],
            },
            cookies=self.request_cookies,
        )

    def bootstrap_search(self, response):
        self.logger.info(
            "Bootstrap completed with status %s, starting at page %s for %s pages",
            response.status,
            self.start_page,
            self.total_pages,
        )
        yield self.build_search_request(self.start_page)

    def build_search_request(self, page):
        return FormRequest(
            url=self.search_url,
            formdata={
                "data[numero]": "",
                "data[anno]": "0",
                "data[page]": str(page),
                "data[t_edicion]": "Cualquiera",
                "data[buscar]": "1",
            },
            headers={
                **self.base_request_headers,
                "Referer": self.bootstrap_url,
            },
            cookies=self.request_cookies,
            callback=self.parse,
            cb_kwargs={"page": page},
            errback=self.handle_request_error,
        )

    def parse(self, response, page=0):
        self.logger.info("Received page %s with status %s", page, response.status)
        results = response.css("div.result-gaceta")

        if not results:
            self.logger.info("No results on page %s", page)
            return

        for result in results:
            # Use the visible-xs block (mobile view) to extract metadata
            # since it appears first and has all the data we need
            block = result.css("div.view-inicio.visible-xs")

            tipo_edicion = block.css(
                "div.views-field-field-tipo-edicion-gaceta "
                "span.field-content::text"
            ).get("").strip()

            fecha = block.css(
                "div.views-field-field-fecha-gaceta "
                "span.date-display-single::text"
            ).get("").strip()

            numero = block.css(
                "div.views-field-field-numero-de-gaceta "
                "span.field-content::text"
            ).get("").strip()

            pdf_url = block.css(
                "div.views-field-field-fichero-gaceta a::attr(href)"
            ).get("")

            # Normas are in the col-sm-7 sibling
            normas = result.css(
                "div.normas-de-la-gaceta div.norma-gaceta a::text"
            ).getall()
            normas = [n.strip() for n in normas if n.strip()]

            item = GacetaOficialItem()
            item["tipo_edicion"] = tipo_edicion
            item["fecha"] = fecha
            item["numero"] = numero
            item["normas"] = normas
            item["pdf_url"] = response.urljoin(pdf_url) if pdf_url else ""
            item["file_urls"] = [response.urljoin(pdf_url)] if pdf_url else []

            yield item

        next_page = page + 1
        last_page = self.start_page + self.total_pages
        if next_page < last_page:
            yield self.build_search_request(next_page)

    def handle_request_error(self, failure):
        request = failure.request
        page = request.cb_kwargs.get("page", "unknown")
        self.logger.warning("Request failed for page %s: %s", page, failure.value)
