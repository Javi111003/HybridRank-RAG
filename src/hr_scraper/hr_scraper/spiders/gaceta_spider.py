import scrapy
from ..items import GacetaItem

class GacetaSpider(scrapy.Spider):
    name = "gaceta"
    allowed_domains = ["gacetaoficial.gob.cu"]
    start_urls = ["https://www.gacetaoficial.gob.cu/es/algunas-legislaciones-cubanas"]

    def parse(self, response):
        rows = response.xpath("//table[contains(@class, 'views-table')]/tbody/tr")
        for row in rows:
            title = row.xpath(".//td[contains(@class, 'views-field-title')]/text()").get()
            pdf_url = row.xpath(".//td[contains(@class, 'views-field-field-fichero-legislacion-cubana')]/a/@href").get()

            if pdf_url:
                item = GacetaItem()
                item["title"] = title.strip() if title else "Sin título"
                item["file_urls"] = [response.urljoin(pdf_url)]
                yield item

        # Paginación
        next_page = response.xpath("//li[@class='pager-next']/a/@href").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)
