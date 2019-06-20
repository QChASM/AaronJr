import json
import os

import Aaron.job as job
import Aaron.options as options
from AaronTools.json_extension import JSONDecoder as ATDecoder
from AaronTools.json_extension import JSONEncoder as ATEncoder


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        Calls appropriate encoding method for supported AaronTools types.
        If type not supported, calls the default `default` method
        """
        if isinstance(obj, options.Theory):
            return self._encode_theory(obj)
        elif isinstance(obj, options.CatalystMetaData):
            return self._encode_metadata(obj)
        elif isinstance(obj, job.Job):
            return self._encode_job(obj)
        elif isinstance(obj, options.ClusterOpts):
            return self._encode_cluster_opts(obj)
        else:
            super().default(obj)

    def _encode_theory(self, obj):
        """
        Encodes the data necessary to re-inialize Theory().
        """
        rv = {"_type": obj.__class__.__name__}
        for step, theory in obj.by_step.items():
            tmp = {}
            tmp["method"] = theory.method
            tmp["basis"] = theory.basis
            tmp["ecp"] = theory.ecp
            tmp["gen_basis"] = theory.gen_basis
            tmp["route_kwargs"] = theory.route_kwargs
            rv[step] = tmp
        return rv

    def _encode_metadata(self, obj):
        rv = {"_type": obj.__class__.__name__}
        for key, val in obj.__dict__.items():
            if key == "catalyst":
                json_location = os.path.join(
                    obj.ts_directory,
                    "json",
                    obj.get_basename().split(".")[-1] + ".json",
                )
                with open(json_location, "w") as f:
                    json.dump(val, f, cls=ATEncoder)
                val = json_location
            rv[key] = val
        return rv

    def _encode_job(self, obj):
        rv = {"_type": "Job"}
        for key, val in obj.__dict__.items():
            rv[key] = val
        return rv
        return obj.__dict__

    def _encode_cluster_opts(self, obj):
        rv = obj.__dict__
        rv["_type"] = "ClusterOpts"
        return rv


class JSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs
        )

    def object_hook(self, obj):
        if "_type" not in obj:
            return obj
        if obj["_type"] == "Theory":
            return self._decode_theory(obj)
        if obj["_type"] == "CatalystMetaData":
            return self._decode_metadata(obj)
        if obj["_type"] == "Job":
            return self._decode_job(obj)
        if obj["_type"] == "ClusterOpts":
            return self._decode_cluster_opts(obj)

    def _decode_theory(self, obj):
        for step, theory in obj.items():
            if step == "_type":
                continue
            tmp = options.Theory()
            tmp.method = theory["method"]
            tmp.basis = theory["basis"]
            tmp.ecp = theory["ecp"]
            tmp.gen_basis = theory["gen_basis"]
            tmp.route_kwargs = theory["route_kwargs"]
            options.Theory.by_step[step] = tmp
        return options.Theory.by_step[0.0]

    def _decode_metadata(self, obj):
        kwargs = {}
        for key, val in obj.items():
            if key == "_type":
                continue
            elif key == "catalyst":
                with open(val) as f:
                    val = json.load(f, cls=ATDecoder)
            elif key in ["substrate_change", "ligand_change"]:
                if val is not None:
                    val = tuple(val)
            kwargs[key] = val
        return options.CatalystMetaData(**kwargs)

    def _decode_job(self, obj):
        rv = job.Job(
            obj["catalyst_data"],
            obj["theory"],
            obj["cluster_opts"],
            step=obj["step"],
        )
        for key, val in obj.items():
            if key in ["_type", "catalyst_data", "theory", "step"]:
                continue
            rv.__dict__[key] = val
        return rv

    def _decode_cluster_opts(self, obj):
        rv = options.ClusterOpts()
        for key, val in obj.items():
            if key in ["_type"]:
                continue
            rv.__dict__[key] = val
        return rv
