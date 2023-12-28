import requests


def get_weather_forecast(
    location: str, url: str = "https://www.jma.go.jp/bosai/forecast/data/forecast/"
) -> list[str]:
    """3日分の天気予測を取得する
    args:
        location str:予測場所のID(東京は130000)
                IDはこちらから確認:https://www.jma.go.jp/bosai/common/const/area.json
        url str:リクエスト先URL
    return:
        3日分の天気予報
    """
    try:
        endpoint = f"{url}{location}.json"
        response = requests.get(endpoint)
        weather_data = response.json()[0]["timeSeries"][0]["areas"][0]["weathers"]
        return [weather.replace("\u3000", " ") for weather in weather_data]
    except requests.RequestException as e:
        raise requests.RequestException(
            f"An error occurred while making the request: {e}"
        )
