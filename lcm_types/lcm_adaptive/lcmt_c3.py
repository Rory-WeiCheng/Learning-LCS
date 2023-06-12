"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class lcmt_c3(object):
    __slots__ = ["utime", "data_size", "data"]

    __typenames__ = ["int64_t", "int32_t", "double"]

    __dimensions__ = [None, None, ["data_size"]]

    def __init__(self):
        self.utime = 0
        self.data_size = 0
        self.data = []

    def encode(self):
        buf = BytesIO()
        buf.write(lcmt_c3._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">qi", self.utime, self.data_size))
        buf.write(struct.pack('>%dd' % self.data_size, *self.data[:self.data_size]))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != lcmt_c3._get_packed_fingerprint():
            raise ValueError("Decode error")
        return lcmt_c3._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = lcmt_c3()
        self.utime, self.data_size = struct.unpack(">qi", buf.read(12))
        self.data = struct.unpack('>%dd' % self.data_size, buf.read(self.data_size * 8))
        return self
    _decode_one = staticmethod(_decode_one)

    def _get_hash_recursive(parents):
        if lcmt_c3 in parents: return 0
        tmphash = (0x1e3728c1fddd3042) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if lcmt_c3._packed_fingerprint is None:
            lcmt_c3._packed_fingerprint = struct.pack(">Q", lcmt_c3._get_hash_recursive([]))
        return lcmt_c3._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

    def get_hash(self):
        """Get the LCM hash of the struct"""
        return struct.unpack(">Q", lcmt_c3._get_packed_fingerprint())[0]

