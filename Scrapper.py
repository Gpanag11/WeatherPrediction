from bs4 import BeautifulSoup
import requests
import time
import random
import csv

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36',
]


def get_html(url):
    """
    Retrieves the HTML content from the specified URL.

    Returns:
        str: HTML content of the page if the request is successful; otherwise, an empty string.

    This function makes an HTTP GET request to the given URL with a randomly chosen user-agent
    from the user_agents list. It handles response status and returns the HTML content for further processing.
    """
    print(f"data from {url}")
    headers = {
        'User-Agent': random.choice(user_agents)
    }
    response = requests.get(url, headers=headers)
    if not response.ok:
        print(f'Error: Code: {response.status_code}, url: {url}')
    else:
        print("response came back successfully!")
    return response.text


def extract_data(html):
    """
    Extracts temperature data from HTML content.

    Args:
        html (str): HTML content of a weather data page.

    Returns:
        list of tuples: Each tuple contains day, temperature during the day, and temperature at night.

    """
    soup = BeautifulSoup(html, 'html.parser')

    days = [day.get_text(strip=True) for day in soup.find_all('td', class_='first')]
    # print("Days extracted:", days)

    temperatures = soup.find_all('td', class_='first_in_group positive')
    temperatures = [temp.get_text(strip=True) for temp in temperatures]
    temperature_day = temperatures[::2]
    temperature_night = temperatures[1::2]

    return list(zip(days, temperature_day, temperature_night))


def main():
    """
    Main function to scrape and save weather data.

    Scrapes temperature data for each day from the start year to the end year, and saves it to a CSV file.
    The data includes the year, month, day, daytime temperature, and nighttime temperature.
    """
    start_year = 2002
    end_year = 2023
    base_url = f'https://www.gismeteo.ru/diary/11901/'
    data = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == end_year and month > 10:  # last month is october 2023
                break
            url = f"{base_url}{year}/{month}/"
            html = get_html(url)
            month_data = extract_data(html)
            for entry in month_data:
                day_data = {
                    "year": year,
                    "month": month,
                    "day": entry[0],
                    "temperature_day": entry[1],
                    "temperature_night": entry[2]
                }
                data.append(day_data)
                print(f"Added data for day {day_data['day']}")
            print(f'Scraping is done for {year}-{month}')
            time.sleep(random.uniform(3, 7))

    with open('weather_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['year', 'month', 'day', 'temperature_day', 'temperature_night']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print("Save data to weather_data.csv")


# run once
if __name__ == "__main__":
    main()
