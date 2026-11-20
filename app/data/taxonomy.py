"""3C/数码品类的领域知识：品牌、价格带、规格生成规则、卖点标签。

数据是合成的，但规则按真实市场来写：
- 品牌只出现在自己该出现的品类里（避免"Apple 显示器"这种穿帮）；
- 规格与价格强相关（旗舰才给 1TB、独显笔记本才上 4060）；
- 卖点标签由规格反推，而不是随机打标，保证评测集和检索结果自洽。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Brand:
    name: str
    series: tuple[str, ...]
    base_price: float
    price_span: tuple[float, float] = (0.6, 1.7)
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Category:
    name: str
    brands: tuple[Brand, ...]
    price_range: tuple[float, float]
    tiers: tuple[float, float, float]  # 低/中/高/旗舰 的分界
    title_suffix: str
    spec_builder: Callable[..., dict[str, str]] = field(repr=False)
    tag_rule: Callable[..., list[str]] = field(repr=False)


COLORS = {
    "手机": ["曜金黑", "雪域白", "星河蓝", "晨曦金", "钛灰", "青雾绿", "月光银"],
    "笔记本电脑": ["深空灰", "星光色", "银灰", "曜石黑", "冰霜银"],
    "平板电脑": ["深空灰", "星光色", "冰霜银", "晨曦金"],
    "无线耳机": ["白色", "黑色", "星空蓝", "香槟金", "云灰"],
    "智能手表": ["午夜黑", "星光银", "钛金属", "玫瑰金", "石墨色"],
    "相机": ["黑色", "银色", "石墨黑"],
    "蓝牙音箱": ["黑色", "白色", "海军蓝", "卡其", "迷彩"],
    "机械键盘": ["黑白拼色", "纯白", "深空灰", "云雾蓝", "奶油黄"],
    "游戏鼠标": ["黑色", "白色", "粉色", "银灰"],
    "显示器": ["黑色", "银白"],
}

VARIANTS = ("", "", "", "Pro", "Pro Max", "Ultra", "Plus", "SE", "青春版", "Mini")
RAM_OPTIONS = (8, 12, 16, 24, 32, 64)
STORAGE_OPTIONS = (128, 256, 512, 1024, 2048)


def _tier(price: float, tiers: tuple[float, float, float]) -> int:
    if price < tiers[0]:
        return 0
    if price < tiers[1]:
        return 1
    if price < tiers[2]:
        return 2
    return 3


# ---------------------------------------------------------------- 手机
def _phone_specs(rng: random.Random, price: float, tier: int) -> dict[str, str]:
    size = rng.choice([6.1, 6.3, 6.5, 6.7, 6.8, 6.9])
    refresh = rng.choices([60, 90, 120, 144], weights=[2 if tier == 0 else 1, 2, 5, 2 if tier >= 2 else 0])[0]
    chip = rng.choice(
        [
            ["天玑7300", "骁龙6 Gen1", "天玑6300"],
            ["骁龙7+ Gen3", "天玑8300", "骁龙7s Gen2"],
            ["骁龙8 Gen3", "天玑9300+", "天玑9200"],
            ["骁龙8 Elite", "天玑9400", "A18 Pro"],
        ][tier]
    )
    ram = rng.choice([8, 12, 16, 24] if tier >= 2 else [8, 12, 16])
    storage = rng.choice([128, 256, 512, 1024] if tier >= 2 else [128, 256, 512])
    battery = rng.choice([4500, 4800, 5000, 5500, 6000])
    charge = rng.choice([33, 67, 80, 100, 120] if tier >= 2 else [18, 33, 67])
    main_cam = rng.choice([5000, 6400, 10800, 20000] if tier >= 2 else [4800, 5000, 6400])
    tele = rng.choice(["潜望式长焦", "3倍光学变焦", "无"]) if tier >= 2 else "无"
    weight = round(rng.uniform(168, 232), 1)
    ip = rng.choice(["IP68", "IP54", "无"])
    return {
        "屏幕": f"{size:.1f}英寸",
        "刷新率": f"{refresh}Hz",
        "处理器": chip,
        "运行内存": f"{ram}GB",
        "机身存储": f"{storage}GB",
        "电池容量": f"{battery}mAh",
        "充电功率": f"{charge}W",
        "后置主摄": f"{main_cam}万像素",
        "长焦": tele,
        "重量": f"{weight}g",
        "防水等级": ip,
    }


def _phone_tags(specs: dict[str, str], price: float, rating: float) -> list[str]:
    tags: list[str] = []
    if int(specs["后置主摄"].replace("万像素", "")) >= 10000 or specs["长焦"].startswith("潜望"):
        tags.append("拍照")
    battery = int(specs["电池容量"].replace("mAh", ""))
    if battery >= 5000:
        tags.append("续航")
    if int(specs["充电功率"].replace("W", "")) >= 67:
        tags.append("快充")
    refresh = int(specs["刷新率"].replace("Hz", ""))
    if refresh >= 120:
        tags.append("高刷")
    if refresh >= 120 and specs["处理器"] in ("骁龙8 Elite", "天玑9400", "A18 Pro", "骁龙8 Gen3"):
        tags.append("游戏")
    if float(specs["重量"].replace("g", "")) <= 185:
        tags.append("轻薄")
    if price >= 5000:
        tags.append("旗舰")
    if price <= 2500 and rating >= 4.4:
        tags.append("性价比")
    if int(specs["运行内存"].replace("GB", "")) >= 16:
        tags.append("大内存")
    return tags


# ---------------------------------------------------------------- 笔记本
def _laptop_specs(rng: random.Random, price: float, tier: int) -> dict[str, str]:
    size = rng.choice([13.3, 14.0, 15.6, 16.0])
    screen = f"{size:.1f}英寸"
    cpu = rng.choice(
        [
            ["酷睿 i5-13420H", "锐龙 R5-7530U", "酷睿 i5-12450H"],
            ["酷睿 i7-13620H", "锐龙 R7-7735HS", "酷睿 Ultra 5"],
            ["酷睿 Ultra 7", "锐龙 R9-8945HS", "酷睿 i9-13900H"],
            ["M4", "M4 Pro", "酷睿 Ultra 9", "锐龙 R9-8945HX"],
        ][tier]
    )
    gpu = rng.choice(
        [["核显"], ["核显", "RTX 4050"], ["RTX 4060", "RTX 4070", "核显"], ["RTX 4070", "RTX 4080", "核显"]][tier]
    )
    ram = rng.choice([16, 32] if tier >= 2 else [8, 16])
    disk = rng.choice([512, 1024, 2048] if tier >= 2 else [512, 1024])
    refresh = rng.choice([60, 120, 144, 165, 240]) if tier >= 2 else rng.choice([60, 90])
    weight = round(rng.uniform(1.15, 2.75), 2)
    return {
        "屏幕": screen,
        "刷新率": f"{refresh}Hz",
        "处理器": cpu,
        "显卡": gpu,
        "内存": f"{ram}GB",
        "硬盘": f"{disk}GB SSD",
        "重量": f"{weight}kg",
        "电池": f"{rng.choice([56, 70, 75, 84, 99])}Wh",
    }


def _laptop_tags(specs: dict[str, str], price: float, rating: float) -> list[str]:
    tags: list[str] = []
    weight = float(specs["重量"].replace("kg", ""))
    if weight <= 1.5:
        tags.append("轻薄")
    if specs["显卡"] != "核显":
        tags.append("游戏")
    if float(specs["重量"].replace("kg", "")) <= 1.6 and price <= 6000:
        tags.append("学生")
    if specs["处理器"] in ("M4", "M4 Pro", "酷睿 Ultra 9"):
        tags.append("办公")
    if price >= 8000:
        tags.append("旗舰")
    if price <= 4000 and rating >= 4.3:
        tags.append("性价比")
    return tags


# ---------------------------------------------------------------- 平板
def _tablet_specs(rng: random.Random, price: float, tier: int) -> dict[str, str]:
    cpu = rng.choice(
        [["天玑9000", "骁龙870"], ["骁龙8+ Gen1", "天玑9200"], ["M2", "骁龙8 Gen3"], ["M4", "M4 Pro"]][tier]
    )
    return {
        "屏幕": f"{rng.choice([10.9, 11.0, 12.1, 12.9, 13.0]):.1f}英寸",
        "刷新率": f"{rng.choice([60, 90, 120, 144])}Hz",
        "处理器": cpu,
        "内存": f"{rng.choice([6, 8, 12, 16])}GB",
        "存储": f"{rng.choice([128, 256, 512, 1024])}GB",
        "重量": f"{round(rng.uniform(440, 700), 0):.0f}g",
        "手写笔": rng.choice(["支持", "支持", "不支持"]),
        "网络": rng.choice(["WiFi版", "5G版"]),
    }


def _tablet_tags(specs: dict[str, str], price: float, rating: float) -> list[str]:
    tags = ["办公"] if specs["手写笔"] == "支持" else ["便携"]
    if int(specs["屏幕"].replace("英寸", "").split(".")[0]) >= 12:
        tags.append("大屏")
    if price >= 5000:
        tags.append("旗舰")
    if price <= 2000:
        tags.append("性价比")
    return tags


# ---------------------------------------------------------------- 耳机
def _headphone_specs(rng: random.Random, price: float, tier: int) -> dict[str, str]:
    kind = rng.choice(["入耳式", "入耳式", "头戴式", "开放式"])
    anc = rng.choice([0, 22, 35, 42, 48]) if tier >= 1 else rng.choice([0, 0, 22])
    life = rng.choice([4, 6, 8, 10, 12, 30, 40])
    return {
        "类型": kind,
        "降噪深度": f"{anc}dB" if anc else "无主动降噪",
        "续航": f"{life}小时",
        "蓝牙版本": rng.choice(["5.2", "5.3", "5.4"]),
        "重量": f"{round(rng.uniform(4, 260), 1)}g",
        "防水": rng.choice(["IPX4", "IPX5", "IP54", "无"]),
    }


def _headphone_tags(specs: dict[str, str], price: float, rating: float) -> list[str]:
    tags: list[str] = []
    if specs["降噪深度"] != "无主动降噪":
        tags.append("降噪")
    if specs["类型"] == "头戴式":
        tags.append("音质")
    if price >= 1500:
        tags.append("旗舰")
    if price <= 400:
        tags.append("性价比")
    if specs["防水"] != "无":
        tags.append("运动")
    return tags


# ---------------------------------------------------------------- 手表
def _watch_specs(rng: random.Random, price: float, tier: int) -> dict[str, str]:
    return {
        "屏幕": f"{rng.choice([1.4, 1.5, 1.6, 1.8, 1.9])}英寸",
        "续航": f"{rng.choice([2, 7, 14, 21])}天",
        "防水": rng.choice(["5ATM", "IP68", "潜水级"]),
        "血氧": rng.choice(["支持", "支持", "不支持"]),
        "ECG": rng.choice(["支持", "不支持"]),
        "GPS": rng.choice(["双频GPS", "单频GPS", "无"]),
        "材质": rng.choice(["铝金属", "不锈钢", "钛金属", "氟橡胶"]),
    }


def _watch_tags(specs: dict[str, str], price: float, rating: float) -> list[str]:
    tags: list[str] = ["运动"]
    if specs["ECG"] == "支持" or specs["血氧"] == "支持":
        tags.append("健康")
    if int(specs["续航"].replace("天", "")) >= 14:
        tags.append("长续航")
    if price >= 2000:
        tags.append("旗舰")
    if price <= 500:
        tags.append("性价比")
    return tags


# ---------------------------------------------------------------- 相机
def _camera_specs(rng: random.Random, price: float, tier: int) -> dict[str, str]:
    sensor = rng.choice(["全画幅", "APS-C", "全画幅", "M43"])
    return {
        "传感器": sensor,
        "像素": f"{rng.choice([2010, 2420, 2610, 3300, 4500, 6100])}万像素",
        "视频": rng.choice(["4K30P", "4K60P", "4K120P", "8K30P"]),
        "防抖": rng.choice(["机身五轴防抖", "镜头防抖", "无"]),
        "重量": f"{rng.choice([360, 429, 530, 658, 720])}g",
        "卡口": rng.choice(["E卡口", "RF卡口", "Z卡口", "X卡口", "M43卡口"]),
    }


def _camera_tags(specs: dict[str, str], price: float, rating: float) -> list[str]:
    tags = ["拍照"]
    if specs["视频"] in ("4K120P", "8K30P"):
        tags.append("视频")
    if specs["传感器"] == "全画幅":
        tags.append("旗舰")
    if price <= 5000:
        tags.append("入门")
    return tags


# ---------------------------------------------------------------- 音箱
def _speaker_specs(rng: random.Random, price: float, tier: int) -> dict[str, str]:
    return {
        "功率": f"{rng.choice([10, 20, 30, 40, 60])}W",
        "单元": f"{rng.choice([1, 2, 3])}单元",
        "蓝牙版本": rng.choice(["5.0", "5.1", "5.3", "5.4"]),
        "续航": f"{rng.choice([8, 12, 20, 24, 30])}小时",
        "防水": rng.choice(["IPX7", "IPX5", "IP67", "无"]),
    }


def _speaker_tags(specs: dict[str, str], price: float, rating: float) -> list[str]:
    tags = ["音质"]
    if int(specs["功率"].replace("W", "")) >= 30:
        tags.append("大功率")
    if specs["防水"] != "无":
        tags.append("户外")
    if price <= 300:
        tags.append("性价比")
    return tags


# ---------------------------------------------------------------- 键盘
def _keyboard_specs(rng: random.Random, price: float, tier: int) -> dict[str, str]:
    return {
        "轴体": rng.choice(["红轴", "茶轴", "青轴", "磁轴", "静音红轴"]),
        "配列": rng.choice(["68键", "75键", "87键", "98键", "104键"]),
        "连接": rng.choice(["三模无线", "有线", "双模无线"]),
        "背光": rng.choice(["RGB背光", "白色背光", "无背光"]),
        "键帽": rng.choice(["PBT键帽", "ABS键帽", "PBT热升华"]),
    }


def _keyboard_tags(specs: dict[str, str], price: float, rating: float) -> list[str]:
    tags = ["办公"]
    if "无线" in specs["连接"]:
        tags.append("无线")
    if specs["轴体"] in ("磁轴", "红轴") and specs["背光"] == "RGB背光":
        tags.append("游戏")
    if "PBT" in specs["键帽"]:
        tags.append("耐用")
    return tags


# ---------------------------------------------------------------- 鼠标
def _mouse_specs(rng: random.Random, price: float, tier: int) -> dict[str, str]:
    return {
        "传感器": rng.choice(["PAW3950", "PAW3395", "HERO 2", "Focus Pro 30K"]),
        "DPI": f"{rng.choice([16000, 26000, 30000, 32000])}",
        "连接": rng.choice(["三模无线", "有线", "双模无线"]),
        "重量": f"{rng.choice([49, 55, 60, 68, 80, 89])}g",
        "回报率": rng.choice(["1000Hz", "4000Hz", "8000Hz"]),
    }


def _mouse_tags(specs: dict[str, str], price: float, rating: float) -> list[str]:
    tags = ["游戏"] if int(specs["DPI"]) >= 26000 else ["办公"]
    if float(specs["重量"].replace("g", "")) <= 60:
        tags.append("轻量化")
    if "无线" in specs["连接"]:
        tags.append("无线")
    return tags


# ---------------------------------------------------------------- 显示器
def _monitor_specs(rng: random.Random, price: float, tier: int) -> dict[str, str]:
    return {
        "尺寸": f"{rng.choice([23.8, 27.0, 31.5, 34.0])}英寸",
        "分辨率": rng.choice(["1920x1080", "2560x1440", "3440x1440", "3840x2160"]),
        "刷新率": f"{rng.choice([60, 75, 144, 165, 240, 360])}Hz",
        "面板": rng.choice(["IPS", "VA", "OLED", "MiniLED"]),
        "接口": rng.choice(["HDMI+DP", "HDMI+DP+Type-C", "Type-C 90W"]),
        "色域": f"{rng.choice([95, 99, 100])}% sRGB",
    }


def _monitor_tags(specs: dict[str, str], price: float, rating: float) -> list[str]:
    tags: list[str] = []
    if int(specs["刷新率"].replace("Hz", "")) >= 144:
        tags.append("游戏")
    if "3840" in specs["分辨率"] or "3440" in specs["分辨率"]:
        tags.append("4K")
    if "Type-C" in specs["接口"]:
        tags.append("办公")
    if "OLED" in specs["面板"] or "MiniLED" in specs["面板"]:
        tags.append("旗舰")
    if price <= 1000:
        tags.append("性价比")
    return tags


def _brands(*items: tuple[str, tuple[str, ...], float, tuple[str, ...]]) -> tuple[Brand, ...]:
    return tuple(Brand(name=n, series=s, base_price=b, aliases=a) for n, s, b, a in items)


CATEGORIES: dict[str, Category] = {
    "手机": Category(
        name="手机",
        brands=_brands(
            ("Apple", ("iPhone",), 7499, ("apple", "苹果")),
            ("华为", ("Mate", "Pura", "nova"), 4999, ("huawei",)),
            ("小米", ("小米", "MIX"), 3299, ("xiaomi", "mi")),
            ("OPPO", ("Find X", "Reno"), 3499, ("oppo",)),
            ("vivo", ("X", "S"), 3299, ("vivo", "维沃")),
            ("荣耀", ("Magic", "数字"), 2999, ("honor",)),
            ("三星", ("Galaxy S", "Galaxy A"), 4599, ("samsung",)),
            ("一加", ("Ace", "数字"), 2799, ("oneplus",)),
            ("realme", ("GT", "数字"), 1799, ("真我",)),
            ("Redmi", ("K", "Note"), 1499, ("红米",)),
        ),
        price_range=(899, 12999),
        tiers=(2000, 3500, 5500),
        title_suffix="5G手机",
        spec_builder=_phone_specs,
        tag_rule=_phone_tags,
    ),
    "笔记本电脑": Category(
        name="笔记本电脑",
        brands=_brands(
            ("Apple", ("MacBook Air", "MacBook Pro"), 9999, ("apple", "苹果")),
            ("联想", ("小新", "拯救者", "ThinkPad"), 5499, ("lenovo",)),
            ("戴尔", ("XPS", "灵越", "游匣"), 5299, ("dell",)),
            ("惠普", ("星Book", "暗影精灵", "战66"), 4799, ("hp",)),
            ("华硕", ("天选", "无畏", "ROG"), 5999, ("asus",)),
            ("华为", ("MateBook",), 5699, ("huawei",)),
            ("小米", ("RedmiBook", "小米笔记本"), 4299, ("xiaomi", "红米")),
            ("机械革命", ("极光", "蛟龙"), 5299, ("mechrevo",)),
            ("微软", ("Surface Laptop", "Surface Pro"), 8999, ("microsoft",)),
            ("宏碁", ("掠夺者", "非凡"), 4599, ("acer",)),
        ),
        price_range=(2999, 24999),
        tiers=(4500, 7000, 12000),
        title_suffix="笔记本电脑",
        spec_builder=_laptop_specs,
        tag_rule=_laptop_tags,
    ),
    "平板电脑": Category(
        name="平板电脑",
        brands=_brands(
            ("Apple", ("iPad Air", "iPad Pro", "iPad"), 4299, ("apple", "苹果")),
            ("华为", ("MatePad", "MatePad Pro"), 2699, ("huawei",)),
            ("小米", ("平板", "平板 Pro"), 1999, ("xiaomi",)),
            ("联想", ("小新Pad", "拯救者Y700"), 1699, ("lenovo",)),
            ("三星", ("Galaxy Tab S", "Galaxy Tab A"), 3299, ("samsung",)),
            ("荣耀", ("平板", "MagicPad"), 1599, ("honor",)),
            ("OPPO", ("Pad",), 1899, ("oppo",)),
            ("vivo", ("Pad",), 1999, ("vivo",)),
        ),
        price_range=(999, 9999),
        tiers=(1800, 3000, 5500),
        title_suffix="平板电脑",
        spec_builder=_tablet_specs,
        tag_rule=_tablet_tags,
    ),
    "无线耳机": Category(
        name="无线耳机",
        brands=_brands(
            ("Apple", ("AirPods", "AirPods Pro", "AirPods Max"), 1399, ("apple", "苹果")),
            ("华为", ("FreeBuds", "FreeBuds Pro"), 899, ("huawei",)),
            ("小米", ("Buds", "Air"), 349, ("xiaomi",)),
            ("索尼", ("WF", "WH"), 1499, ("sony",)),
            ("Bose", ("QuietComfort", "Ultra"), 2299, ("bose",)),
            ("漫步者", ("LolliPods", "NeoBuds", "W820"), 399, ("edifier",)),
            ("森海塞尔", ("Momentum", "CX"), 1299, ("sennheiser",)),
            ("三星", ("Galaxy Buds",), 999, ("samsung",)),
            ("OPPO", ("Enco",), 499, ("oppo",)),
            ("韶音", ("OpenFit", "OpenRun"), 1099, ("shokz",)),
        ),
        price_range=(99, 4999),
        tiers=(300, 800, 1800),
        title_suffix="无线蓝牙耳机",
        spec_builder=_headphone_specs,
        tag_rule=_headphone_tags,
    ),
    "智能手表": Category(
        name="智能手表",
        brands=_brands(
            ("Apple", ("Watch Series", "Watch Ultra"), 2999, ("apple", "苹果")),
            ("华为", ("Watch GT", "Watch"), 1488, ("huawei",)),
            ("小米", ("Watch", "手环"), 499, ("xiaomi",)),
            ("佳明", ("Forerunner", "Fenix"), 2999, ("garmin",)),
            ("三星", ("Galaxy Watch",), 1999, ("samsung",)),
            ("OPPO", ("Watch",), 1299, ("oppo",)),
            ("荣耀", ("Watch",), 899, ("honor",)),
            ("华米", ("Amazfit",), 799, ("amazfit",)),
        ),
        price_range=(199, 6999),
        tiers=(600, 1500, 3000),
        title_suffix="智能手表",
        spec_builder=_watch_specs,
        tag_rule=_watch_tags,
    ),
    "相机": Category(
        name="相机",
        brands=_brands(
            ("索尼", ("A7", "A6700", "ZV-E"), 10999, ("sony",)),
            ("佳能", ("EOS R", "EOS RP"), 8999, ("canon",)),
            ("尼康", ("Z5", "Z6", "Z50"), 8499, ("nikon",)),
            ("富士", ("X-T", "X-S", "X100"), 8999, ("fujifilm",)),
            ("松下", ("Lumix",), 7999, ("panasonic",)),
            ("徕卡", ("Q", "D-Lux"), 19999, ("leica",)),
            ("奥林巴斯", ("OM",), 6999, ("olympus",)),
            ("大疆", ("Pocket", "Osmo"), 2499, ("dji",)),
            ("GoPro", ("HERO",), 2499, ("gopro",)),
        ),
        price_range=(1699, 45999),
        tiers=(4000, 9000, 15000),
        title_suffix="微单相机",
        spec_builder=_camera_specs,
        tag_rule=_camera_tags,
    ),
    "蓝牙音箱": Category(
        name="蓝牙音箱",
        brands=_brands(
            ("JBL", ("Flip", "Charge", "PartyBox"), 899, ("jbl",)),
            ("漫步者", ("M230", "R1700", "花再"), 399, ("edifier",)),
            ("哈曼卡顿", ("琉璃", "音乐行星"), 1299, ("harman",)),
            ("索尼", ("SRS", "ULT"), 999, ("sony",)),
            ("Bose", ("SoundLink", "Home"), 1999, ("bose",)),
            ("小度", ("智能音箱",), 299, ("xiaodu",)),
            ("小米", ("小爱", "音箱"), 249, ("xiaomi",)),
            ("华为", ("Sound",), 799, ("huawei",)),
        ),
        price_range=(99, 6999),
        tiers=(300, 900, 2000),
        title_suffix="蓝牙音箱",
        spec_builder=_speaker_specs,
        tag_rule=_speaker_tags,
    ),
    "机械键盘": Category(
        name="机械键盘",
        brands=_brands(
            ("罗技", ("K", "MX"), 599, ("logitech",)),
            ("雷蛇", ("黑寡妇", "猎魂光蛛"), 899, ("razer",)),
            ("樱桃", ("MX",), 799, ("cherry",)),
            ("达尔优", ("A", "EK"), 299, ("dareu",)),
            ("杜伽", ("K",), 599, ("ducky",)),
            ("阿米洛", ("VA",), 499, ("varmilo",)),
            ("京东京造", ("J",), 199, ("京造",)),
            ("黑爵", ("AK",), 189, ("ajazz",)),
        ),
        price_range=(89, 2999),
        tiers=(250, 600, 1200),
        title_suffix="机械键盘",
        spec_builder=_keyboard_specs,
        tag_rule=_keyboard_tags,
    ),
    "游戏鼠标": Category(
        name="游戏鼠标",
        brands=_brands(
            ("罗技", ("G", "MX"), 449, ("logitech",)),
            ("雷蛇", ("毒蝰", "炼狱蝰蛇"), 499, ("razer",)),
            ("卓威", ("EC", "S"), 699, ("zowie",)),
            ("达尔优", ("EM", "A"), 199, ("dareu",)),
            ("赛睿", ("Aerox", "Prime"), 599, ("steelseries",)),
            ("雷柏", ("VT",), 149, ("rapoo",)),
            ("微软", ("Pro",), 499, ("microsoft",)),
            ("黑爵", ("AJ",), 129, ("ajazz",)),
        ),
        price_range=(49, 1999),
        tiers=(200, 450, 800),
        title_suffix="游戏鼠标",
        spec_builder=_mouse_specs,
        tag_rule=_mouse_tags,
    ),
    "显示器": Category(
        name="显示器",
        brands=_brands(
            ("戴尔", ("U", "AW"), 2499, ("dell",)),
            ("华硕", ("ROG", "ProArt", "TUF"), 2299, ("asus",)),
            ("小米", ("Redmi", "显示器"), 899, ("xiaomi",)),
            ("AOC", ("Q", "AG"), 1299, ("aoc",)),
            ("明基", ("PD", "MOBIUZ"), 2299, ("benq",)),
            ("LG", ("UltraGear", "UltraFine"), 2299, ("lg",)),
            ("三星", ("奥德赛", "ViewFinity"), 2599, ("samsung",)),
            ("雷神", ("Z",), 1199, ("thunderobot",)),
            ("HKC", ("G", "T"), 999, ("hkc",)),
        ),
        price_range=(499, 12999),
        tiers=(1200, 2500, 5000),
        title_suffix="显示器",
        spec_builder=_monitor_specs,
        tag_rule=_monitor_tags,
    ),
}

CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "手机": ("手机", "智能机", "phone", "拍照手机", "游戏手机"),
    "笔记本电脑": ("笔记本", "笔记本电脑", "laptop", "轻薄本", "游戏本", "电脑"),
    "平板电脑": ("平板", "平板电脑", "ipad", "pad"),
    "无线耳机": ("耳机", "蓝牙耳机", "无线耳机", "降噪耳机", "耳麦"),
    "智能手表": ("手表", "智能手表", "watch", "手环"),
    "相机": ("相机", "微单", "单反", "摄影", "相机机身"),
    "蓝牙音箱": ("音箱", "音响", "蓝牙音箱", "智能音箱"),
    "机械键盘": ("键盘", "机械键盘", "keyboard"),
    "游戏鼠标": ("鼠标", "游戏鼠标", "mouse"),
    "显示器": ("显示器", "屏幕", "电竞屏", "monitor"),
}

# 卖点标签 → 文案片段，用于商品描述生成
SELLING_POINTS: dict[str, str] = {
    "拍照": "大底主摄配合长焦，人像和夜景直出可用",
    "视频": "支持高帧率视频录制，Vlog 与短片创作更从容",
    "续航": "大电池加持，重度使用也能撑满一天",
    "快充": "高功率快充，洗漱的功夫回血大半",
    "高刷": "高刷新率屏幕，滑动与游戏画面更跟手",
    "游戏": "性能释放激进，长时间游戏帧率稳定",
    "轻薄": "整机重量控制出色，通勤携带无负担",
    "降噪": "深度主动降噪，地铁通勤也能安静听歌",
    "音质": "调音偏流行，人声与低频层次清楚",
    "办公": "接口与续航均衡，适合日常办公场景",
    "学生": "价格与性能平衡，网课与作业够用",
    "性价比": "同价位配置给得足，预算有限也能拿下",
    "旗舰": "用料与做工对标旗舰，质感在线",
    "大内存": "大内存多开不卡，后台应用留存率高",
    "防水": "支持生活防水，雨天与运动场景更安心",
    "运动": "佩戴稳固，运动时不易脱落",
    "健康": "支持心率血氧监测，健康数据一目了然",
    "长续航": "续航按周计算，出差不用带充电器",
    "户外": "防尘防水等级高，户外露营可直接用",
    "大功率": "大功率输出，聚会氛围拉满",
    "无线": "无线连接稳定，桌面更清爽",
    "轻量化": "轻量化设计，长时间握持不累手",
    "耐用": "键帽耐磨，长期使用不打油",
    "4K": "高分屏细节丰富，修图与观影都舒服",
    "大屏": "大尺寸屏幕，追剧与办公更沉浸",
    "入门": "入门价位给到完整功能，适合新手",
    "便携": "体积小巧，随手放进包里",
    "高性能": "性能释放充分，多任务处理流畅",
}
