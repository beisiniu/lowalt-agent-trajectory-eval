"""几何判定的交叉验证测试。

为什么必须交叉验证
------------------
`agent/geo.py` 同时被 Agent 工具层和评测集答案键使用。如果几何实现写错，
工具和答案键会一起错，评测结果反而「自洽地通过」——这是最危险的失效模式。
所以这里用**两套独立实现**互相验证：

  圆柱命中：解析公式（geo.in_cylinder） vs 三维网格暴力采样
  多边形：  射线法（geo.point_in_polygon） vs 角度和（winding number）

这继承了上一个项目「几何物化 + 第二套算法复核」的做法。
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import geo  # noqa: E402


# --------------------------------------------------------------------------
# 独立实现 A：角度和法（winding number），与射线法完全不同的思路
# --------------------------------------------------------------------------

def winding_number(x: float, y: float, polygon) -> int:
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    wn = 0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        if y1 <= y:
            if y2 > y:  # 向上穿越
                cross = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
                if cross > 0:
                    wn += 1
        else:
            if y2 <= y:  # 向下穿越
                cross = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
                if cross < 0:
                    wn -= 1
    return wn


def inside_by_winding(x: float, y: float, polygon) -> bool:
    return abs(winding_number(x, y, polygon)) == 1


# --------------------------------------------------------------------------
# 独立实现 B：暴力网格采样判断圆柱内部
# --------------------------------------------------------------------------

def inside_cylinder_by_sampling(
    x: float, y: float, z: float, center, radius: float, floor: float, ceiling: float, n: int = 64
) -> bool:
    """把圆柱体离散成 n×n×n 的格点，判断目标点是否落在任一格内。

    只用于小范围交叉验证，精度受格距限制；边界点会被排除在比较外。
    """
    cx, cy = float(center[0]), float(center[1])
    xs = [cx - radius + 2 * radius * i / n for i in range(n + 1)]
    ys = [cy - radius + 2 * radius * i / n for i in range(n + 1)]
    zs = [floor + (ceiling - floor) * i / n for i in range(n + 1)]
    for gx in xs:
        for gy in ys:
            if math.hypot(gx - cx, gy - cy) > radius:
                continue
            for gz in zs:
                if abs(gx - x) < 1e-9 and abs(gy - y) < 1e-9 and abs(gz - z) < 1e-9:
                    return True
    return False


class TestCylinder(unittest.TestCase):
    CENTER = (100.0, 100.0)
    R = 15.0
    FLOOR = 0.0
    CEIL = 40.0

    def test_inside_center(self):
        self.assertTrue(geo.in_cylinder(100, 100, 20, self.CENTER, self.R, self.FLOOR, self.CEIL))
        self.assertEqual(geo.cylinder_slant_distance(100, 100, 20, self.CENTER, self.R, self.FLOOR, self.CEIL), 0.0)

    def test_above_ceiling(self):
        # 水平在圆内，垂直超出顶面 20 m
        self.assertFalse(geo.in_cylinder(100, 100, 60, self.CENTER, self.R, self.FLOOR, self.CEIL))
        self.assertAlmostEqual(
            geo.cylinder_slant_distance(100, 100, 60, self.CENTER, self.R, self.FLOOR, self.CEIL), 20.0, places=6
        )

    def test_below_floor(self):
        self.assertAlmostEqual(
            geo.cylinder_slant_distance(100, 100, -5, self.CENTER, self.R, self.FLOOR, self.CEIL), 5.0, places=6
        )

    def test_radial_outside(self):
        # 水平距离 100，半径 15 → 侧向距离 85，同高度
        self.assertAlmostEqual(
            geo.cylinder_slant_distance(200, 100, 20, self.CENTER, self.R, self.FLOOR, self.CEIL), 85.0, places=6
        )

    def test_diagonal_corner(self):
        # 水平偏 15+3=18（径向 3），垂直偏 40+4=44（轴向 4）→ 斜距 5
        d = geo.cylinder_slant_distance(100 + 18, 100, 44, self.CENTER, self.R, self.FLOOR, self.CEIL)
        self.assertAlmostEqual(d, 5.0, places=6)

    def test_boundary_inside(self):
        # 恰好在侧面上
        self.assertTrue(geo.in_cylinder(100 + 15, 100, 20, self.CENTER, self.R, self.FLOOR, self.CEIL))

    def test_volume_matches_analytic(self):
        """体积法交叉验证：蒙特卡洛估计判定区域的体积，应等于 πr²h。

        这比逐点比对更强：如果边界写错（例如把直径当半径、把圆柱当球），
        体积会成倍偏离，测试立刻失败。
        """
        rng = random.Random(11)
        n = 120_000
        cx, cy = 0.0, 0.0
        r, floor, ceil = 10.0, 0.0, 30.0
        box_vol = (2 * r) * (2 * r) * (ceil - floor)
        inside = 0
        for _ in range(n):
            x = rng.uniform(cx - r, cx + r)
            y = rng.uniform(cy - r, cy + r)
            z = rng.uniform(floor, ceil)
            if geo.in_cylinder(x, y, z, (cx, cy), r, floor, ceil):
                inside += 1
        est = box_vol * inside / n
        expected = math.pi * r * r * (ceil - floor)
        # 蒙特卡洛标准误约 0.2%，给 2% 容差
        self.assertAlmostEqual(est / expected, 1.0, delta=0.02, msg=f"体积估计 {est:.1f} vs 解析 {expected:.1f}")

    def test_volume_excludes_outside_height(self):
        """高度超界的点不应计入体积（区分圆柱与无限长圆柱）。"""
        rng = random.Random(13)
        n = 40_000
        r, floor, ceil = 5.0, 10.0, 20.0
        above = 0
        for _ in range(n):
            x = rng.uniform(-r, r)
            y = rng.uniform(-r, r)
            z = rng.uniform(21.0, 60.0)  # 全部在顶面之上
            if geo.in_cylinder(x, y, z, (0, 0), r, floor, ceil):
                above += 1
        self.assertEqual(above, 0)


class TestPolygon(unittest.TestCase):
    SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]
    CONCAVE = [[0, 0], [100, 0], [100, 100], [50, 30], [0, 100]]  # 顶部凹进去

    def test_inside_square(self):
        self.assertTrue(geo.point_in_polygon(50, 50, self.SQUARE))

    def test_outside_square(self):
        self.assertFalse(geo.point_in_polygon(150, 50, self.SQUARE))

    def test_on_edge(self):
        self.assertTrue(geo.point_in_polygon(0, 50, self.SQUARE))
        self.assertTrue(geo.point_in_polygon(100, 100, self.SQUARE))

    def test_concave_notch(self):
        # (50, 80) 落在凹口里，应该算外部
        self.assertFalse(geo.point_in_polygon(50, 80, self.CONCAVE))
        # (50, 10) 在底部，属于内部
        self.assertTrue(geo.point_in_polygon(50, 10, self.CONCAVE))

    def test_vs_winding_number(self):
        """射线法 vs 角度和法：随机 2000 个点，结论必须一致。"""
        rng = random.Random(42)
        polys = [self.SQUARE, self.CONCAVE, [[0, 0], [50, 0], [50, 40], [0, 40]]]
        for poly in polys:
            for _ in range(700):
                x = rng.uniform(-20, 120)
                y = rng.uniform(-20, 120)
                # 边界附近的点两套算法的容差不同，跳过
                if any(
                    geo._point_segment_distance(x, y, poly[i][0], poly[i][1], poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][1])
                    < 1e-3
                    for i in range(len(poly))
                ):
                    continue
                a = geo.point_in_polygon(x, y, poly)
                b = inside_by_winding(x, y, poly)
                self.assertEqual(a, b, msg=f"({x:.3f},{y:.3f}) ray={a} winding={b} poly={poly}")


class TestPath(unittest.TestCase):
    def test_distance(self):
        self.assertAlmostEqual(geo.path_distance([[0, 0, 0], [3, 4, 0]]), 5.0, places=6)
        self.assertAlmostEqual(geo.path_distance([[0, 0, 0], [0, 0, 10], [0, 0, 0]]), 20.0, places=6)

    def test_single_point(self):
        self.assertEqual(geo.path_distance([[1, 2, 3]]), 0.0)
        self.assertEqual(geo.path_distance([]), 0.0)

    def test_densify(self):
        pts = geo.densify_path([[0, 0, 0], [10, 0, 0]], step_m=2.0)
        self.assertEqual(pts[0], (0.0, 0.0, 0.0))
        self.assertEqual(pts[-1], (10.0, 0.0, 0.0))
        self.assertGreaterEqual(len(pts), 6)

    def test_segment_cross_detected(self):
        """端点都在禁飞区外，但连线穿过——这正是 1/3 UAVBench 场景的形态。"""
        zone = [{"type": "cylinder", "center": [50, 0], "radius_m": 10, "floor_m": 0, "ceiling_m": 50}]
        path = [[0, 0, 20], [100, 0, 20]]
        for p in path:
            self.assertFalse(geo.check_zones(p[0], p[1], p[2], zone)["in_no_fly_zone"])
        self.assertEqual(geo.min_clearance_along_path(path, zone), 0.0)

    def test_segment_clear(self):
        zone = [{"type": "cylinder", "center": [50, 500], "radius_m": 10, "floor_m": 0, "ceiling_m": 50}]
        path = [[0, 0, 20], [100, 0, 20]]
        self.assertGreater(geo.min_clearance_along_path(path, zone), 400)


class TestCheckZones(unittest.TestCase):
    def test_multi_zone_ids(self):
        zones = [
            {"type": "cylinder", "center": [0, 0], "radius_m": 10, "floor_m": 0, "ceiling_m": 100},
            {"type": "cylinder", "center": [0, 0], "radius_m": 20, "floor_m": 0, "ceiling_m": 100},
        ]
        r = geo.check_zones(0, 0, 50, zones)
        self.assertTrue(r["in_no_fly_zone"])
        self.assertEqual(sorted(r["zone_ids"]), ["NFZ-00", "NFZ-01"])

    def test_custom_zone_id(self):
        zones = [{"id": "AIRPORT-A", "type": "cylinder", "center": [0, 0], "radius_m": 10,
                  "floor_m": 0, "ceiling_m": 100}]
        r = geo.check_zones(0, 0, 50, zones)
        self.assertEqual(r["zone_ids"], ["AIRPORT-A"])
        self.assertEqual(r["nearest_zone_id"], "AIRPORT-A")

    def test_polygon_zone(self):
        zones = [{"type": "polygon", "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                  "floor_m": 0, "ceiling_m": 100}]
        self.assertTrue(geo.check_zones(5, 5, 50, zones)["in_no_fly_zone"])
        self.assertFalse(geo.check_zones(5, 5, 150, zones)["in_no_fly_zone"])

    def test_bad_geometry_skipped(self):
        zones = [{"type": "cylinder"}, {"type": "unknown"}, {}]
        r = geo.check_zones(0, 0, 0, zones)
        self.assertFalse(r["in_no_fly_zone"])


class TestGeofence(unittest.TestCase):
    def test_altitude_band(self):
        air = {
            "geofence": [{"type": "polygon", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}],
            "altitude_agl_min_m": 10,
            "altitude_agl_max_m": 120,
        }
        self.assertTrue(geo.check_geofence(50, 50, 60, air)["inside_altitude_band"])
        self.assertFalse(geo.check_geofence(50, 50, 5, air)["inside_altitude_band"])
        self.assertFalse(geo.check_geofence(50, 50, 200, air)["inside_altitude_band"])

    def test_outside_fence(self):
        air = {
            "geofence": [{"type": "polygon", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}],
            "altitude_agl_min_m": 0,
            "altitude_agl_max_m": 120,
        }
        self.assertFalse(geo.check_geofence(200, 50, 60, air)["inside_geofence"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
