from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import requests
import time

url = 'https://www.investing.com/economic-calendar/'

# Set up Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# Path to chromedriver
# Make sure this path is correct and matches where you extracted chromedriver.exe
chromedriver_path = 'C:\\chromedriver\\chromedriver.exe'

service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Navigate to the desired page
    driver.get(url)
    
    # Sleep for a few seconds to allow the page to load
    time.sleep(5)  # Adjust as needed

    # Get the page source and pass it to BeautifulSoup
    page_source = driver.page_source
    
    # Parse the content using BeautifulSoup
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')

    # Find the table containing calendar events
    calendar_table = soup.find('table', class_='economicCalendarTable')

    if not calendar_table:
        print("Table not found on the page.")
        exit()

    # Extract rows from the table
    rows = calendar_table.find_all('tr')[1:]  # Skip the header row

    news = []

    for row in rows:
        try:
            time = row.find('td', class_='first left').text.strip()
            currency = row.find('td', class_='left').text.strip()
            event = row.find_all('td', class_='left')[1].text.strip()
            actual = row.find('td', class_='act').text.strip()
            forecast = row.find('td', class_='fore').text.strip()
            previous = row.find('td', class_='prev').text.strip()

            news_item = {
                'time': time,
                'currency': currency,
                'event': event,
                'actual': actual,
                'forecast': forecast,
                'previous': previous,
            }
            news.append(news_item)
        except AttributeError:
            continue

    # Print extracted news
    for item in news:
        print(f"Time: {item['time']}")
        print(f"Currency: {item['currency']}")
        print(f"Event: {item['event']}")
        print(f"Actual: {item['actual']}")
        print(f"Forecast: {item['forecast']}")
        print(f"Previous: {item['previous']}\n")
finally:
    driver.quit()