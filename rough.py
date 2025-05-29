from selenium import webdriver
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import json

query = 'us university cs rankings'

# Set up the Chrome WebDriver
driver = uc.Chrome(use_subprocess=True)

# URL of the THE World University Rankings page
# url = 'https://www.google.com/'
url = 'https://chat.openai.com/'

# Load the page
driver.get(url)

# Allow time for JavaScript to render
time.sleep(5)

prompt = "What is reinforcement learning?"

text_area = driver.find_element(By.XPATH, '//p[@class="placeholder"]')
text_area.click()
text_area.send_keys(prompt)
text_area.send_keys(Keys.RETURN)

time.sleep(10)

prev_text = ""
while True:
    response_divs = driver.find_elements(By.CLASS_NAME, "markdown")  # Get all responses
    if response_divs:
        response_text = response_divs[-1].text  # Get the latest response
        if response_text == prev_text:  # Check if response stopped updating
            break
        prev_text = response_text
    time.sleep(1)
print("\nChatGPT Response:\n", response_text)

# response_divs = driver.find_elements(By.CLASS_NAME, "markdown")  # ChatGPT response class
# if response_divs:
#     response_text = response_divs[-1].text  # Get the last response
#     print("\nChatGPT Response:\n", response_text)
# else:
#     print("Could not capture response.")

# search_box = driver.find_element(By.NAME, 'q')
# search_box.send_keys(query)
# search_box.send_keys(Keys.RETURN)

# time.sleep(5)

# search_results = driver.find_elements(By.CSS_SELECTOR, "h3")[:10]

# links = []
# for result in search_results:
#     try:
#         link = result.find_element(By.XPATH, "./ancestor::a").get_attribute("href")
#         links.append(link)
#     except:
#         continue  # Skip if no link is found

# content = {}

# for idx, link in enumerate(links, start=1):
#     driver.find_element(By.TAG_NAME,'body').send_keys(Keys.COMMAND + 't') 
#     driver.get(link)
#     time.sleep(5)  # Wait for the page to load
    
#     soup = BeautifulSoup(driver.page_source, "html.parser")

#     # Remove unwanted elements (nav bars, sidebars, footers, scripts, etc.)
#     for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
#         tag.extract()

#     # Extract meaningful text
#     page_text = soup.get_text(separator="\n").strip()

#     driver.find_element(By.TAG_NAME,'body').send_keys(Keys.COMMAND + 't')
#     content[link] = page_text[:500]
    
# with open('content.json', 'w') as f:
#     json.dump(content, f)

# Print the top 5 links
# for idx, link in enumerate(links, start=1):
#     print(f"{idx}. {link}")

# # Accept cookies if prompted
# try:
#     accept_cookies_button = driver.find_element(By.XPATH, '//button[text()="Accept cookies"]')
#     accept_cookies_button.click()
#     time.sleep(2)
# except:
#     pass

# # Extract university rankings
# universities = []
# rows = driver.find_elements(By.XPATH, '//table[contains(@class, "ranking-table")]//tbody//tr')

# for row in rows:
#     rank = row.find_element(By.XPATH, './/td[contains(@class, "rank")]').text
#     name = row.find_element(By.XPATH, './/td[contains(@class, "name")]//a').text
#     location = row.find_element(By.XPATH, './/td[contains(@class, "location")]').text
#     score = row.find_element(By.XPATH, './/td[contains(@class, "score")]').text
#     universities.append([rank, name, location, score])

# Close the driver
driver.quit()

# # Create a DataFrame and save to CSV
# df = pd.DataFrame(universities, columns=['Rank', 'University', 'Location', 'Score'])
# df.to_csv('THE_World_University_Rankings_2023.csv', index=False)

# print("Data has been successfully scraped and saved to 'THE_Rankings_2025.csv'.")
