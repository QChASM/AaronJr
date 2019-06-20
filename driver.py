from fireworks.user_objects.firetasks.templatewriter_task import FiretaskBase


class ArchiveDirTask(FiretaskBase):
    """
   Wrapper around shutil.make_archive to make tar archives.

   Args:
       base_name (str): Name of the file to create.
       format (str): Optional. one of "zip", "tar", "bztar" or "gztar".
   """

    _fw_name = "ArchiveDirTask"
    required_params = ["base_name"]
    optional_params = ["format"]

    def run_task(self, fw_spec):
        shutil.make_archive(
            self["base_name"], format=self.get("format", "gztar"), root_dir="."
        )


class RemoteGaussianJob(FiretaskBase):
    """

    """

    _fw_name = "RemoteGaussianJob"
    required_params = ["com_file", "comp_opts", "executable_location"]

    def run_task(self, fw_spec):
        pass
