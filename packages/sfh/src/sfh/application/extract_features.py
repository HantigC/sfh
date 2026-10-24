from dataclasses import dataclass

from sfh.domain.entities import Descriptors, FeatureConfig, Keypoints
from sfh.domain.errors import DatasetNotFoundError
from sfh.domain.ports.features import FeatureExtractor, ImageLoader
from sfh.domain.ports.progress import NullProgress, ProgressReporter
from sfh.domain.ports.unit_of_work import UnitOfWorkFactory


@dataclass(frozen=True, slots=True)
class ExtractFeaturesResult:
    config: FeatureConfig
    extracted: int
    skipped: int  # already had results for this config


class ExtractFeatures:
    """Detect keypoints and compute descriptors for every image of a dataset.

    Results are keyed by the extractor's (name, params), so several methods can coexist,
    and images that already have results are skipped, which makes the step resumable.
    """

    def __init__(
        self, uow: UnitOfWorkFactory, loader: ImageLoader, extractor: FeatureExtractor
    ) -> None:
        self._uow = uow
        self._loader = loader
        self._extractor = extractor

    def __call__(
        self, dataset_name: str, progress: ProgressReporter | None = None
    ) -> ExtractFeaturesResult:
        progress = progress or NullProgress()
        with self._uow() as uow:
            dataset = uow.datasets.get_by_name(dataset_name)
            if dataset is None:
                raise DatasetNotFoundError(dataset_name)
            config = uow.configs.get_or_create(self._extractor.name, self._extractor.params)
            images = uow.images.list_by_dataset(dataset.id)
            uow.commit()

        extracted = skipped = 0
        progress.start(len(images))
        for image in images:
            # One transaction per image: an interrupted run keeps the finished images.
            with self._uow() as uow:
                if uow.keypoints.exists(image.id, config.id):
                    skipped += 1
                else:
                    features = self._extractor.extract(self._loader.load(image.path))
                    uow.keypoints.save(
                        Keypoints(
                            image_id=image.id,
                            config_id=config.id,
                            xy=features.xy,
                            scores=features.scores,
                            scales=features.scales,
                            orientations=features.orientations,
                        )
                    )
                    uow.descriptors.save(Descriptors(image.id, config.id, features.descriptors))
                    uow.commit()
                    extracted += 1
            progress.advance()
        progress.finish()
        return ExtractFeaturesResult(config=config, extracted=extracted, skipped=skipped)
