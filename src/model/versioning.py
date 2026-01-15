from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VersionManager:
    """
    Simple version manager using a text file that stores the last model version.
    """

    def __init__(self, version_file: str):
        self.version_path = Path(version_file)

    def get_last_version(self) -> int:
        """
        Returns last known version, or 0 if file does not exist.
        """
        if not self.version_path.exists():
            logger.info("Version file %s not found. Assuming version 0.", self.version_path)
            return 0
        try:
            content = self.version_path.read_text().strip()
            return int(content)
        except Exception as e:
            logger.error("Error reading version file %s: %s. Assuming version 0.",
                         self.version_path, e)
            return 0

    def set_version(self, version: int):
        """
        Writes the given version number to file.
        """
        self.version_path.write_text(str(version))
        logger.info("Updated version file %s to %d", self.version_path, version)
