from app.core.text import normalize, parse_price, tokenize


def test_normalize_fullwidth_and_spaces():
    assert normalize("Ｈｕａｗｅｉ　 Mate  60") == "huawei mate 60"


def test_tokenize_chinese_query():
    tokens = tokenize("5000以内拍照好的手机")
    assert "拍照" in tokens
    assert "手机" in tokens


def test_tokenize_drops_stopwords():
    tokens = tokenize("我想要一台好用的手机")
    assert "我" not in tokens
    assert "手机" in tokens


def test_parse_price_units():
    assert parse_price("预算5000") == 5000
    assert parse_price("5k左右") == 5000
    assert parse_price("1万以内") == 10000
    assert parse_price("￥4,999.00") == 4999.0
    assert parse_price("没有价格") is None
