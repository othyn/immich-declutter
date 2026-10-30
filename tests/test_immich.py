import pytest

from immich import Immich


class Resp:
    def __init__(self, data, status=200):
        self.data, self.status_code, self.text = data, status, ""

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


def client(permissions):
    im = Immich("http://i", "k")
    routes = {"/api-keys/me": {"permissions": permissions}, "/search/metadata": {"assets": {"items": []}}}
    im.session.request = lambda method, url, **kw: Resp(routes[url.removeprefix(im.base)])
    return im


def test_check_access_reports_missing_permission():
    im = client(["asset.read", "asset.view", "tag.create"])
    with pytest.raises(PermissionError, match="tag.asset"):
        im.check_access()


@pytest.mark.parametrize("perms", [["all"], ["asset.read", "asset.view", "tag.create", "tag.asset"]])
def test_check_access_ok(perms):
    client(perms).check_access()
