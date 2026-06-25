import zstandard as zstd
import pyarrow as pa


class ZstdStreamReader:
    def __init__(self, filepath, read_size=16384):  # Default 16KB
        self.file = open(filepath, 'rb')
        self.decompressor = zstd.ZstdDecompressor().stream_reader(self.file, read_size=read_size)
        self._closed = False

    def read(self, size=-1):
        if self._closed:
            raise ValueError("I/O operation on closed file")
        return self.decompressor.read(size)

    def readable(self):
        return True

    def seekable(self):
        return False

    def writable(self):
        return False

    @property
    def closed(self):
        return self._closed

    def close(self):
        if not self._closed:
            self.decompressor.close()
            self.file.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def arrow_stream_generator_multi(filepaths, read_size=16384, rank=0):
    for filepath in filepaths:
        print(f"[Rank {rank}] Processing file: {filepath}")
        try:
            with ZstdStreamReader(filepath, read_size=read_size) as stream:
                reader = pa.ipc.open_stream(stream)
                for batch in reader:
                    batch_dict = batch.to_pylist()
                    for record in batch_dict:
                        yield record
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue