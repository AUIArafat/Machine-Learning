import urllib

from selenium import webdriver
import requests
from bs4 import BeautifulSoup as BS
import re

url = 'https://www.prothomalo.com/'
agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
page = requests.get(url, headers=agent)
soup = BS(page.content, 'lxml')

for string in soup.stripped_strings:
    if re.search('[a-zA-Z]', string) == None:
        print(repr(string))
# Firefox session
driver = webdriver.Chrome(executable_path='C:/Users/Asus/OneDrive/Desktop/chromedriver.exe')
driver.get(url)
driver.implicitly_wait(100)
main_links = driver.find_element_by_link_text("৩ বছর আগে বাড়ি থেকে বের হয়েছিলেন ছেলে সন্ধান")
print(main_links)
# while True:
#     # navigate to link
#     button = driver.find_elements_by_class_name("link_overlay")
#     button.click()