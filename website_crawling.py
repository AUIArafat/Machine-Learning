import sys
import time
import urllib

from selenium import webdriver
import requests
from bs4 import BeautifulSoup as BS
import re

for i in range(17220, 80000):
    url = 'https://www.prothomalo.com/sports/article/'+str(i)
    agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    page = requests.get(url, headers=agent).text
    soup = BS(page, "html.parser")
    try:
        title = soup.find("title").get_text()
        body = soup.find("noscript",{"id":"ari-noscript"}).get_text()
        file_name = "D:/Dataset/ProthomAlo/prothomalo_bd_" + str(i) + ".txt"
        a = 0
        for x in range(0, 3):
            a = a + 1
            b = ("Loading" + "." * a + str(i))
            # \r prints a carriage return first, so `b` is printed on top of the previous line.
            sys.stdout.write('\r' + b)
            time.sleep(0.5)
        with open(file_name, "w", encoding="utf-8") as text_file:
            text_file.write(str(title))
            text_file.write("\n")
            text_file.write("\n")
            text_file.write(str(body))
    except:
        print("\nNoting Found")
# print(soup)
# Firefox session
# driver = webdriver.Chrome(executable_path='C:/Users/Asus/OneDrive/Desktop/chromedriver.exe')
# driver.get(url)
# driver.implicitly_wait(100)
# main_links = driver.find_element_by_link_text("৩ বছর আগে বাড়ি থেকে বের হয়েছিলেন ছেলে সন্ধান")
# print(main_links)
# # while True:
# #     # navigate to link
# #     button = driver.find_elements_by_class_name("link_overlay")
# #     button.click()