import requests
import time

# url = "https://www.flipkart.com/twowheelers-at-store?pageUID=1784119661629"
# url = "https://www.naukri.com/python-jobs?k=python&experience=4&wfhType=2"
url = "https://www.geeksforgeeks.org/python/implementing-web-scraping-python-beautiful-soup/"


sessions = requests.Session()


r = requests.get(url)
time.sleep(2)
# print(r)

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

r = sessions.get(url, headers=HEADERS)
print("Status:", r.status_code)
print("Final URL:", r.url)
print("Content-Type:", r.headers.get("Content-Type"))
print(r.text[:500])


with open("new_file.html", "w") as f:
    f.write(r.text)