import os
from dotenv import load_dotenv
load_dotenv()
import httpx
import json

def search_city(city: str):
    url = os.getenv("GET_CITY_URL")
    params = {
        "location":city
    }
    headers = {
        "X-QW-Api-Key":os.getenv("GET_WEATHER")
    }
    with httpx.Client() as client:
            response = client.get(url = url, params=params,headers=headers)
            if response.status_code == 200:
                data = response.json()
                return json.dumps(data,ensure_ascii=False)
            else :
                print("请求异常")
    return

def get_weather(city:str):
    """
    查询具体城市的实时天气情况
    :param city: 城市名称
    """
    response1 = search_city(city)
    response = json.loads(response1)
    city = response["location"][0]
    lat = float(city["lat"])
    lon = float(city["lon"])
    url = f"https://n76mthb3jn.re.qweatherapi.com/weather/v1/current/{lat}/{lon}"
    headers = {
        "X-QW-Api-Key":os.getenv("GET_WEATHER")
    }
    with httpx.Client() as client:
        resp = client.get(url = url,headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            return json.dumps(data,ensure_ascii=False)
        else :
            print("请求出错")
            return

# 测试demo
if __name__ == "__main__":
    result = search_city("哈尔滨")
    print(result)

    result2 = get_weather("哈尔滨")
    print(result2)
