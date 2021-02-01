import requests
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

static_headers = {
    "Content-Type": "application/json;charset=UTF-8",
    "User-Agent": "Mozilla/5.0 (Linux; Android 9; MI 6 Build/PKQ1.190118.001; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/78.0.3904.62 XWEB/2693 MMWEBSDK/200701 Mobile Safari/537.36 MMWEBID/7120 MicroMessenger/7.0.17.1720(0x2700113F) Process/appbrand0 WeChat/arm64 NetType/WIFI Language/zh_CN ABI/arm64",
    "Referer":"https://servicewechat.com/wx8227f55dc4490f45/77/page-frame.html"
}

json = """{"key":"eyJuYW1lIjoi6ZKf5Lyf5rCRIiwieGgiOiIyMDE3MjEzMDc0IiwieGIiOiLnlLciLCJvcGVuaWQiOiJvSWFJSTBVMEMtY0pkYmNuYmhVTVBlbUlsUE80IiwieHhkeiI6IuW4guWFrOi3r+WxgOaXgSIsInN6ZHEiOiLmuZbljZfnnIEs5aiE5bqV5biCLOWohOaYn+WMuiIsImxvY2F0aW9uQmlnIjoi5Lit5Zu9LOa5luWNl+ecgSzlqITlupXluIIs5aiE5pif5Yy6IiwibG9jYXRpb25TbWFsbCI6Iua5luWNl+ecgeWohOW6leW4guWohOaYn+WMuumdkuaYpeihlyIsImxhdGl0dWRlIjoyNy43MjYsImxvbmdpdHVkZSI6MTExLjk5MzEsInl3amNxemJsIjoi5L2O6aOO6ZmpIiwieXdqY2hibGoiOiLml6AiLCJ4anpkeXdxemJsIjoi5pegIiwidHdzZnpjIjoi5pivIiwieXd5dGR6eiI6IuaXoCIsImJlaXpodSI6IuaXoCIsIm1yZGtrZXkiOiJqUWtCT0theCIsInRpbWVzdGFtcCI6MTYxMjE0NjAwM30="}"""

url = "https://we.cqu.pt/api/mrdk/post_mrdk_info.php"

response = requests.post(url = url,json=json,headers = static_headers,verify=False)

print(response.raw)

