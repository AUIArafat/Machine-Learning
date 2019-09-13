import requests
from bs4 import BeautifulSoup
import os
import csv
import tqdm

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36',
    'Referer': 'https://freesound.org/'
}

login_data = {
    'username': 'valid freesound user name',
    'password': 'account password'
}
domain = 'https://freesound.org'
login_url = 'https://freesound.org/home/login/'


def get_links_of_all_sound(search_link):
    """
    :param search_link: freesound url
    :return: returns tuple of 2 list (audio links and license) in the url
    """
    lists_of_sound_links = []
    lists_of_sound_license = []
    with requests.Session() as s:
        r = s.get(search_link, headers=headers)
        soup = BeautifulSoup(r.content, 'html.parser')
        mydivs_1 = soup.findAll("div", {"class": "sample_player_small"})
        for divs in mydivs_1:
            x = divs.findAll('img', {'title': ["This sound is licensed under the Creative Commons Attribution "
                                               "license.", "This sound is public domain."]})
            if len(x) > 0:
                mn = divs.findAll("a", {"class": "title"})
                lists_of_sound_links.append(domain + mn[0].get('href'))
                lists_of_sound_license.append(x[0]['alt'])
    return lists_of_sound_links, lists_of_sound_license


def get_downloadable_links(list_of_sounds_links):
    """
    :param list_of_sounds_links: audio URL list
    :return: returns tuple of downloadable file links and their name
    """
    my_links_info = []
    lists_of_download = []
    file_name = []
    with requests.Session() as s:
        response_login = s.get(login_url, headers=headers)
        soup = BeautifulSoup(response_login.content, 'html.parser')
        login_data['csrfmiddlewaretoken'] = soup.find('input', attrs={'name': 'csrfmiddlewaretoken'})['value']
        s.post(login_url, data=login_data, headers=headers)
        for sites in list_of_sounds_links:
            response = s.get(sites, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            link = soup.find("a", {"id": "download_button"})
            my_links_info.append(link)
        for divs in my_links_info:
            lists_of_download.append(domain+divs.get('href'))
            file_name.append(divs.get('href').split('/')[-1])
    return lists_of_download, file_name


def download_audio(download_url, category):
    """
    downloads audio from given url in specified category
    :param download_url: audio download url
    :param category: audio category
    """
    path = './data/'
    save_path = os.path.join(path, category)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = download_url.split('/')[-1]
    print(file_name)
    with requests.Session() as s:
        r = s.get(login_url, headers=headers)
        soup = BeautifulSoup(r.content, 'html.parser')
        login_data['csrfmiddlewaretoken'] = soup.find('input', attrs={'name': 'csrfmiddlewaretoken'})['value']
        s.post(login_url, data=login_data, headers=headers)
        response = s.get(download_url, stream=True)
        print(response.status_code)
        with open(os.path.join(save_path, file_name), 'wb') as f:
            for chunk in response.iter_content(chunk_size = 1024*1024):
                if chunk:
                    f.write(chunk)


def create_csv_file(a, b, c, d, e):
    with open('combined_file.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerows(zip(a, b, c, d, e))


if __name__ == '__main__':
    with open('test_files_list.txt') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            final_download_files = []
            category = row[0]
            page_link = row[1]
            end_page = row[2]
            for i in range(1, (int(end_page) + 1)):
                category_list = []
                down_link = page_link + str(i) + '#sound'
                print(down_link)
                list_of_sounds_per_page, list_of_license_info = get_links_of_all_sound(down_link)
                list_of_downloadable_audio, list_of_file_name = get_downloadable_links(list_of_sounds_per_page)
                for items in list_of_downloadable_audio:
                    final_download_files.append(items)
                    category_list.append(category)
                create_csv_file(list_of_file_name, list_of_license_info, list_of_downloadable_audio,
                                list_of_sounds_per_page, category_list)
            print('Downloadable total files : ', len(final_download_files))
            for files in tqdm.tqdm(final_download_files):
                download_audio(files, category)
