# 添加在 SimulationConfig 之前
@dataclass
class CrackSpec:
    """单条裂缝的规格"""

    center_x: float = 0.300
    center_y: float = 0.150
    width: float = 0.120  # 裂缝长度
    height: float = 0.010  # 裂缝开度
    orientation: str = "horizontal"  # horizontal, vertical, angled
    angle_deg: float = 30.0  # 仅 angled 时使用
    material_name: str = "free_space"  # free_space 或 water_fill

    def to_dict(self) -> dict:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "width": self.width,
            "height": self.height,
            "orientation": self.orientation,
            "angle_deg": self.angle_deg,
            "material_name": self.material_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CrackSpec":
        return cls(**data)

    @property
    def crack_angle_deg(self) -> float:
        if self.orientation == "vertical":
            return 90.0
        if self.orientation == "angled":
            return self.angle_deg
        return 0.0

    def size_x(self) -> float:
        if self.orientation == "vertical":
            return self.height
        return self.width

    def size_y(self) -> float:
        if self.orientation == "vertical":
            return self.width
        return self.height

    def corners_xy(self) -> List[Tuple[float, float]]:
        """返回裂缝四个角的坐标，用于绘制和边界计算"""
        half_length = 0.5 * self.width
        half_opening = 0.5 * self.height
        theta = math.radians(self.crack_angle_deg)
        ux = (math.cos(theta), math.sin(theta))
        uy = (-math.sin(theta), math.cos(theta))
        return [
            (
                self.center_x - half_length * ux[0] - half_opening * uy[0],
                self.center_y - half_length * ux[1] - half_opening * uy[1],
            ),
            (
                self.center_x + half_length * ux[0] - half_opening * uy[0],
                self.center_y + half_length * ux[1] - half_opening * uy[1],
            ),
            (
                self.center_x + half_length * ux[0] + half_opening * uy[0],
                self.center_y + half_length * ux[1] + half_opening * uy[1],
            ),
            (
                self.center_x - half_length * ux[0] + half_opening * uy[0],
                self.center_y - half_length * ux[1] + half_opening * uy[1],
            ),
        ]

    def bounds(self) -> Tuple[float, float, float, float]:
        """返回 (x_min, x_max, y_min, y_max)"""
        corners = self.corners_xy()
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        return (min(xs), max(xs), min(ys), max(ys))

    def min_dimension(self) -> float:
        """返回最小尺寸，用于分辨率检查"""
        return min(self.width, self.height)

    def input_lines(self, dz: float, material_map: Dict[str, str]) -> List[str]:
        """生成 gprMax 输入命令"""
        mat_name = material_map.get(self.material_name, self.material_name)

        if self.orientation == "angled":
            # 斜裂缝用两个三角形
            p1, p2, p3, p4 = self.corners_xy()
            return [
                "#triangle: {0:.3f} {1:.3f} 0 {2:.3f} {3:.3f} 0 {4:.3f} {5:.3f} 0 {6:.3f} {7}".format(
                    p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], dz, mat_name
                ),
                "#triangle: {0:.3f} {1:.3f} 0 {2:.3f} {3:.3f} 0 {4:.3f} {5:.3f} 0 {6:.3f} {7}".format(
                    p1[0], p1[1], p3[0], p3[1], p4[0], p4[1], dz, mat_name
                ),
            ]
        else:
            # 水平和竖直裂缝用 box
            x_min, x_max, y_min, y_max = self.bounds()
            return [
                "#box: {0:.3f} {1:.3f} 0 {2:.3f} {3:.3f} {4:.3f} {5}".format(
                    x_min, y_min, x_max, y_max, dz, mat_name
                )
            ]
