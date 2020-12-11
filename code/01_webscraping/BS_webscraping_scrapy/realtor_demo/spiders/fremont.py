import scrapy


class FremontSpider(scrapy.Spider):
    name = 'fremont'

    start_urls = [
        'https://www.realtor.com/realestateandhomes-detail/5505-Farina-Ln_Fremont_CA_94538_M29108-42114']

    def parse(self, response):
        links = response.xpath('//section/div/div[contains(@class, "desc")]')
        yield {"Links": links.getall()}
