class SfhError(Exception):
    """Base class for errors entrypoints are expected to report to the user."""


class DatasetNotFoundError(SfhError):
    def __init__(self, name: str) -> None:
        super().__init__(f"dataset not found: {name!r}")
        self.name = name


class DatasetAlreadyExistsError(SfhError):
    def __init__(self, name: str) -> None:
        super().__init__(f"dataset already exists: {name!r}")
        self.name = name


class ImageNotFoundError(SfhError):
    def __init__(self, image_id: int) -> None:
        super().__init__(f"image not found: {image_id}")
        self.image_id = image_id


class DuplicateImageNameError(SfhError):
    def __init__(self, name: str) -> None:
        super().__init__(f"two images share the name {name!r} within one dataset")
        self.name = name


class ImageReadError(SfhError):
    def __init__(self, path: str) -> None:
        super().__init__(f"could not read image: {path}")
        self.path = path
