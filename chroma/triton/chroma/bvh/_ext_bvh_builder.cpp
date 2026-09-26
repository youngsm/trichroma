#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <bvh/v2/default_builder.h>
#include <bvh/v2/node.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/bbox.h>
#include <bvh/v2/vec.h>

#include <vector>
#include <string>

namespace py = pybind11;

using Scalar = float;
static constexpr size_t Dim = 3;
using Vec3 = bvh::v2::Vec<Scalar, Dim>;
using BBox = bvh::v2::BBox<Scalar, Dim>;
using Node = bvh::v2::Node<Scalar, Dim>;
using Bvh = bvh::v2::Bvh<Node>;

namespace {

inline bvh::v2::DefaultBuilder<Node>::Quality parse_quality(const std::string &quality) {
    if (quality == "low") return bvh::v2::DefaultBuilder<Node>::Quality::Low;
    if (quality == "medium") return bvh::v2::DefaultBuilder<Node>::Quality::Medium;
    return bvh::v2::DefaultBuilder<Node>::Quality::High;
}

} // namespace

py::tuple build_bvh(py::array_t<float, py::array::c_style | py::array::forcecast> vertices,
                    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> triangles,
                    const std::string &quality_str) {
    auto vinfo = vertices.request();
    auto tinfo = triangles.request();

    if (vinfo.ndim != 2 || vinfo.shape[1] != 3)
        throw std::runtime_error("vertices must be a Nx3 array");
    if (tinfo.ndim != 2 || tinfo.shape[1] != 3)
        throw std::runtime_error("triangles must be a Nx3 array");

    const size_t nverts = static_cast<size_t>(vinfo.shape[0]);
    const size_t ntris = static_cast<size_t>(tinfo.shape[0]);

    const auto *verts = static_cast<const float*>(vinfo.ptr);
    const auto *tris = static_cast<const uint32_t*>(tinfo.ptr);

    std::vector<BBox> bboxes(ntris);
    std::vector<Vec3> centers(ntris);

    for (size_t i = 0; i < ntris; ++i) {
        uint32_t i0 = tris[3 * i + 0];
        uint32_t i1 = tris[3 * i + 1];
        uint32_t i2 = tris[3 * i + 2];
        if (i0 >= nverts || i1 >= nverts || i2 >= nverts)
            throw std::runtime_error("triangle index out of range");

        Vec3 v0(verts[3 * i0 + 0], verts[3 * i0 + 1], verts[3 * i0 + 2]);
        Vec3 v1(verts[3 * i1 + 0], verts[3 * i1 + 1], verts[3 * i1 + 2]);
        Vec3 v2(verts[3 * i2 + 0], verts[3 * i2 + 1], verts[3 * i2 + 2]);

        BBox bbox;
        bbox.extend(v0);
        bbox.extend(v1);
        bbox.extend(v2);
        bboxes[i] = bbox;

        centers[i] = (v0 + v1 + v2) * Scalar(1.0f / 3.0f);
    }

    bvh::v2::DefaultBuilder<Node>::Config config;
    config.quality = parse_quality(quality_str);
    config.min_leaf_size = 1;
    config.max_leaf_size = 1;

    auto bvh = bvh::v2::DefaultBuilder<Node>::build(bboxes, centers, config);

    const py::ssize_t nnodes = static_cast<py::ssize_t>(bvh.nodes.size());
    py::array_t<Scalar> bounds({nnodes, static_cast<py::ssize_t>(Dim * 2)});
    py::array_t<uint32_t> first_ids({nnodes});
    py::array_t<uint32_t> prim_counts({nnodes});
    py::array_t<uint32_t> prim_ids({static_cast<py::ssize_t>(bvh.prim_ids.size())});

    auto bounds_mut = bounds.mutable_unchecked<2>();
    auto first_mut = first_ids.mutable_unchecked<1>();
    auto count_mut = prim_counts.mutable_unchecked<1>();

    for (size_t i = 0; i < nnodes; ++i) {
        const auto &node = bvh.nodes[i];
        auto bbox = node.get_bbox();
        for (size_t axis = 0; axis < Dim; ++axis) {
            bounds_mut(i, axis * 2 + 0) = bbox.min[axis];
            bounds_mut(i, axis * 2 + 1) = bbox.max[axis];
        }
        first_mut(i) = static_cast<uint32_t>(node.index.first_id());
        count_mut(i) = static_cast<uint32_t>(node.index.prim_count());
    }

    auto prim_mut = prim_ids.mutable_unchecked<1>();
    for (size_t i = 0; i < bvh.prim_ids.size(); ++i)
        prim_mut(i) = static_cast<uint32_t>(bvh.prim_ids[i]);

    return py::make_tuple(bounds, first_ids, prim_counts, prim_ids);
}

PYBIND11_MODULE(_ext_bvh_builder, m) {
    m.doc() = "Bindings to build BVHs using the ext/bvh library";
    m.def("build", &build_bvh, py::arg("vertices"), py::arg("triangles"), py::arg("quality") = "high");
}
