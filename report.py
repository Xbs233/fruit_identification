import requests
from hyper.contrib import HTTP20Adapter

url = "https://getman.cn/api/request"

json = """{"key":"eyJuYW1lIjoi6ZKf5Lyf5rCRIiwieGgiOiIyMDE3MjEzMDc0IiwieGIiOiLnlLciLCJvcGVuaWQiOiJvSWFJSTBVMEMtY0pkYmNuYmhVTVBlbUlsUE80IiwieHhkeiI6IuW4guWFrOi3r+WxgOaXgSIsInN6ZHEiOiLmuZbljZfnnIEs5aiE5bqV5biCLOWohOaYn+WMuiIsInl3amNxemJsIjoi5L2O6aOO6ZmpIiwieXdqY2hibGoiOiLml6AiLCJ4anpkeXdxemJsIjoi5pegIiwidHdzZnpjIjoi5pivIiwieXd5dGR6eiI6IuaXoCIsImJlaXpodSI6IuaXoCIsIm1yZGtrZXkiOiJqUWtCX0trNCIsInRpbWVzdGFtcCI6MTYxMjE0ODcwNH0="}"""

requests = requests.Session()
requests.mount(url,HTTP20Adapter())
static_headers = {
":authority": "getman.cn",
":method": "POST",
":path": "/api/request",
":scheme": "https",
"accept": "*/*",
"accept-encoding": "gzip, deflate, br",
"accept-language": "zh-CN,zh;q=0.9",
"content-length": "1034",
"content-type": "application/x-www-form-urlencoded; charset=UTF-8",
"cookie": "Hm_lvt_5355caba3fc9511e407d548c7c066f64=1612150075; Hm_lpvt_5355caba3fc9511e407d548c7c066f64=1612150075",
"origin": "https://getman.cn",
"referer": "https://getman.cn/?s=2f0ce8542c7fcfc6b59454159705fe1c",
"sec-fetch-dest":"empty",
"sec-fetch-mode": "cors",
"sec-fetch-site": "same-origin",
"user-agent": """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36""",
"x-client-id": "2c95132d3dff847e6100b47369d76b55",
"x-request-id": "6c4878d2fd4de920fb2e733ba949ddf2",
"x-requested-with": "XMLHttpRequest",
}

content = {
"url": "https://we.cqu.pt/api/mrdk/post_mrdk_info.php",
"method": "POST",
"body":"""{"key":"eyJuYW1lIjoi6ZKf5Lyf5rCRIiwieGgiOiIyMDE3MjEzMDc0IiwieGIiOiLnlLciLCJvcGVuaWQiOiJvSWFJSTBVMEMtY0pkYmNuYmhVTVBlbUlsUE80IiwieHhkeiI6IuW4guWFrOi3r+WxgOaXgSIsInN6ZHEiOiLmuZbljZfnnIEs5aiE5bqV5biCLOWohOaYn+WMuiIsInl3amNxemJsIjoi5L2O6aOO6ZmpIiwieXdqY2hibGoiOiLml6AiLCJ4anpkeXdxemJsIjoi5pegIiwidHdzZnpjIjoi5pivIiwieXd5dGR6eiI6IuaXoCIsImJlaXpodSI6IuaXoCIsIm1yZGtrZXkiOiJqUWtCX0trNCIsInRpbWVzdGFtcCI6MTYxMjE0ODcwNH0="}""",
"header[Content-Type]": "application/json;charset=UTF-8",
"header[User-Agent]": """Mozilla/5.0 (Linux; Android 9; MI 6 Build/PKQ1.190118.001; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/78.0.3904.62 XWEB/2693 MMWEBSDK/200701 Mobile Safari/537.36 MMWEBID/7120 MicroMessenger/7.0.17.1720(0x2700113F) Process/appbrand0 WeChat/arm64 NetType/WIFI Language/zh_CN ABI/arm64""",
"header[Referer]": "https://servicewechat.com/wx8227f55dc4490f45/77/page-frame.html",
}

response = requests.post(url = url,data=content,headers = static_headers)

print(response.text)
