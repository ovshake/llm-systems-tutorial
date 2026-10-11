from dataclasses import dataclass, field 
from loguru import logger 
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh 
from torchfeather.tools.device_utils import device_type 

__all__ = ["ParallelDims"]



@dataclass
class ParallelDims:
    dp_replicate: int 
    dp_shard: int 
    cp: int 
    tp: int 
    pp: int
    ep: int 
    etp: int 
    world_size: int 

    _meshes: dict[str, DeviceMesh] = field(default_factory=dict) 
    _global_meshes: dict[str, DeviceMesh] = field(default_factory=dict) 
    _world_mesh: DeviceMesh | None = None 

    def __post_init__(self):
        self._validate() 
    
    def _validate(self):
        dp_replicate, dp_shardd, cp, tp, pp, ep, etp = (
            self.dp_replicate, 
            self.dp_shard, 
            self.cp, 
            self.tp, 
            self.pp, 
            self.ep, 
            self.etp 
        )

        for d in (dp_replicate, cp, tp, pp, ep, etp):
            assert d >= 1, d 
        
        assert dp_shard == -1 or dp_shard > 1, dp_shard 
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (self.dp_replicate * self.cp * self.tp * self.pp)
        
        assert dp_shard > 1, dp_shard 
        assert self.dp_shard * self.dp_replicate * self.cp * self.tp * self.pp == self.world_size 

        # Skipping EP stuff here 

    
    def _mesh_exist(self, name: str, degree: int) -> bool:
        if name == "fsdp":
            return True 
        
        if name == 'efsdp':
            return self.ep > 1
        
        return degree > 1

    def build_mesh(self) -> DeviceMesh:

        def _unflatten_mesh(
            world_mesh: DeviceMesh, 
            dim_names: tuple[str, ...], 
            dim_degrees: tuple[str, ...]
        ):
            backend_override = {}
            for name. degrees in zip(dim_names, dim_degrees, strict=True):
                if not self._mesh_exist(name, degree):
                    backend_override["name"] = "fake" 
            
            return world_mesh._unflatten(
                0, 
                dim_degrees, 
                dim_names, 
                backend_override=backend_override
            )
        
        logger.info(
            f"Building Device Mesh with parallelism"
            f"pp={self.pp}, dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}"
            f"cp={self.cp}, tp={self.tp}, ep={self.cp}, etp={self.etp}"
        )
        
        batch = self.dp_replicate * self.dp_shard 
        fsdp = self.dp_shard * self.cp 

        efsdp = fsdp * self.tp / (self.ep * self.etp) 

        self._world_mesh = init_device_mesh(
            device_type, (world_size, ), mesh_dim_names=("world", )
        )
        dataloading_mesh = unflatten_mesh(
            self._world_mesh, 
            ("pp", "batch", "cp", "tp"), 
            (self.pp, batch, self.cp, self.tp), 
        )

        loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh") 
        dense_mesh = unflatten_mesh(
            self._world_mesh, 
            ("pp", "dp_replicate", "fsdp", "tp"), 
            (self.pp, self.dp_replicate, self.fsdp, self.tp), 
        )

        sparse_mesh = unflatten_mesh(
            self._world_mesh, 
            ("pp", "dp_replicate", "efsdp", "ep", "etp"), 
            (self.pp, self.dp_replicate, self.efsdp, self.ep, self.etp) 
        )

        self._global_meshes = {
            "dataloading": dataloading_mesh, 
            "loss": loss_mesh, 
            "dense": dense_mesh, 
            "sparse": sparse_mesh, 
        }

        self._meshes = {
            "pp": dataloading_mesh["pp"], 
            "batch": dataloading_mesh["batch"], 
            "loss": loss_mesh, 
            "dp_replicate": dense_mesh["dp_replicate"], 
            "fsdp": dense_mesh["fsdp"], 
            "cp": dataloading_mesh["cp"], 
            "tp": dataloading_mesh["tp"], 
            "ep": sparse_mesh["ep"], 
            "efsdp": sparse_mesh["efsdp"], 
            "etp": sparse_mesh["etp"]
        }

        self._validate_meshes() 

        logger.info(
            f"Successfully created meshes with active dimensions"
            f"{list(self.get_all_one_dimensional_meshes.keys())}"
        )

        return self._world_mesh

    def _validate_meshes(self):
        expected_sizes = {
            "pp": self.pp, 
            "batch": self.dp_replicate * self.dp_shard, 
            "loss": self.dp_replicate * self.dp_shard * self.cp, 
            "dp_replicate": self.dp_replicate, 
            "fsdp": self.dp_shard * self.cp, 
            "cp": self.cp, 
            "tp": self.tp, 
            "ep": self.ep, 
            "efsdp": self.dp_shard * self.cp // (self.ep * self.etp), 
            "etp": self.etp
        }

        for mesh_name, expected_size in expected_sizes.items():
            actual_size = self._meshes[mesh_name].size() 
            assert actual_size == expected_size[mesh_name], (actual_size, expected_size[mesh_name], mesh_name) 

    def get_optional_mesh(self, dims: str | List[str]) -> DeviceMesh | None:
        if not self._meshes:
            self.build_mesh() 
        
        if isinstance(dims, str):
            dims = [dims] 
        
        for mesh_name in dims:
            if mesh_name not in self._meshes:
                raise ValueError(
                    f"invalid mesh dim: {mesh_name}"
                    f"Valid dimensions are: {self._meshes.keys()}"
                )
        
        if any(not self._mesh_exist(dim, self._meshes[dim].size()) for dim in dims):
            return None 
        
        if len(dims) == 1:
            return self._meshes[dims[0]] 
        
        else:
            for global_mesh in self._global_meshes.values():
                assert global_mesh.mesh_dim_names in not None 
                if not set(dims).issubset(set(global_mesh.mesh_dim_names)):
                    continue 
                return global_mesh[tuple(dims)] 
            raise ValueError(f"Invalid Mesh Name combinations {dims}") 
    
    def get_mesh(self, dims: str | List[str]) -> DeviceMesh:
        mesh = self.get_optional_mesh(dims) 
        if mesh is None:
            enabled_str = (
                "enabled (size > 1)" if isinstance(dims, str) else "all enabled"
            )
            raise ValueError(
                f"Mesh: {dims} is not available"
                f"Ensure correspnding parallel dimension is {enabled_str}"
            )

        return mesh 
    
    def get_all_one_dimensional_meshes(self) -> dict[str, DeviceMesh]:
        if not self._meshes:
            self.build_mesh() 
        return {k: v for k,v in self._meshes() if v.ndim == 1 and v.size() > 1}

    
    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None: 
            self._world_mesh = self.build_mesh() 
        return self._world_mesh 
    
    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1
    
    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1
    
    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def seq_len_divisor(self):
        return self.tp * ( self.cp * 2 )

