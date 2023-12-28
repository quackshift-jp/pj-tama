import requests


def get_weather_predict(
    location: str, url: str = "https://www.jma.go.jp/bosai/forecast/data/forecast/"
) -> list[str]:
    """3日分の天気予測を取得する
    args:
    location str:予測場所のID
        ex)
            ID参考:https://www.jma.go.jp/bosai/common/const/area.json
    url str:リクエスト先URL
    """
    endpoint = f"{url}{location}.json"
    response = requests.get(endpoint)
    if response.status_code == 200:
        weather_predict = response.json()[0]["timeSeries"][0]["areas"][0]["weathers"]
        return [weather.replace("\u3000", "") for weather in weather_predict]
