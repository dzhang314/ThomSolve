module ThomSolve

using MultiFloats: rsqrt_r

################################################################# COULOMB ENERGY


export coulomb_energy, coulomb_forces!


function coulomb_energy(points::AbstractMatrix{T}) where {T}
    point_axis = axes(points, 1)
    xyz_axis = axes(points, 2)
    @assert length(xyz_axis) == 3
    x, y, z = xyz_axis
    result = zero(T)
    @inbounds begin
        for i = first(point_axis):(last(point_axis)-1)
            x_i = points[i, x]
            y_i = points[i, y]
            z_i = points[i, z]
            @simd ivdep for j = (i+1):last(point_axis)
                x_j = points[j, x]
                y_j = points[j, y]
                z_j = points[j, z]
                dx = x_i - x_j
                dy = y_i - y_j
                dz = z_i - z_j
                result += rsqrt_r(abs2(dx) + abs2(dy) + abs2(dz))
            end
        end
    end
    return result
end


function coulomb_forces!(
    forces::AbstractMatrix{T},
    point_energies::AbstractVector{T},
    points::AbstractMatrix{T},
) where {T}
    point_axis = axes(forces, 1)
    xyz_axis = axes(forces, 2)
    @assert point_axis == axes(point_energies, 1)
    @assert point_axis == axes(points, 1)
    @assert xyz_axis == axes(points, 2)
    @assert length(xyz_axis) == 3
    x, y, z = xyz_axis
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _half = inv(_two)
    fill!(forces, _zero)
    fill!(point_energies, _zero)
    @inbounds begin
        for i = first(point_axis):(last(point_axis)-1)
            x_i = points[i, x]
            y_i = points[i, y]
            z_i = points[i, z]
            fx_i = _zero
            fy_i = _zero
            fz_i = _zero
            point_energy_i = _zero
            @simd ivdep for j = (i+1):last(point_axis)
                x_j = points[j, x]
                y_j = points[j, y]
                z_j = points[j, z]
                dx = x_i - x_j
                dy = y_i - y_j
                dz = z_i - z_j
                inv_r = rsqrt_r(abs2(dx) + abs2(dy) + abs2(dz))
                point_energy_i += inv_r
                point_energies[j] += inv_r
                inv_r3 = abs2(inv_r) * inv_r
                fx_i = muladd(dx, inv_r3, fx_i)
                fy_i = muladd(dy, inv_r3, fy_i)
                fz_i = muladd(dz, inv_r3, fz_i)
                forces[j, x] = muladd(-dx, inv_r3, forces[j, x])
                forces[j, y] = muladd(-dy, inv_r3, forces[j, y])
                forces[j, z] = muladd(-dz, inv_r3, forces[j, z])
            end
            forces[i, x] += fx_i
            forces[i, y] += fy_i
            forces[i, z] += fz_i
            point_energies[i] += point_energy_i
        end
        @simd ivdep for i in point_axis
            point_energies[i] *= _half
        end
    end
    return forces
end


################################################################# LINEAR ALGEBRA


export orthogonalize_columns!, symmetric_update!


function orthogonalize_columns!(A::AbstractMatrix{T}) where {T}
    _zero = zero(T)
    row_axis = axes(A, 1)
    col_axis = axes(A, 2)
    @inbounds begin
        for j in col_axis
            for k = first(col_axis):(j-1)
                overlap = _zero
                @simd ivdep for i in row_axis
                    overlap = muladd(conj(A[i, k]), A[i, j], overlap)
                end
                @simd ivdep for i in row_axis
                    A[i, j] = muladd(-overlap, A[i, k], A[i, j])
                end
            end
            norm2 = _zero
            @simd ivdep for i in row_axis
                norm2 += abs2(A[i, j])
            end
            inv_norm = rsqrt_r(norm2)
            @simd ivdep for i in row_axis
                A[i, j] *= inv_norm
            end
        end
    end
    return A
end


function symmetric_update!(C::AbstractMatrix{T}, A::AbstractMatrix{T}) where {T}
    # C is assumed to be symmetric; only its lower triangle (i >= j) is updated.
    indices = axes(C, 1)
    @assert axes(C, 2) == indices
    @assert axes(A, 1) == indices
    @inbounds begin
        for k in axes(A, 2)
            for j in indices
                a_jk = A[j, k]
                @simd ivdep for i = j:last(indices)
                    C[i, j] = muladd(A[i, k], a_jk, C[i, j])
                end
            end
        end
    end
    return C
end


################################################################ SPHERE GEOMETRY


export sphere_project!, stereographic_coordinates!, sphere_step!,
    sphere_tangent!, sphere_killing_vectors!


function sphere_project!(points::AbstractMatrix{T}) where {T}
    point_axis = axes(points, 1)
    xyz_axis = axes(points, 2)
    @assert length(xyz_axis) == 3
    x, y, z = xyz_axis
    @inbounds begin
        @simd ivdep for i in point_axis
            x_i = points[i, x]
            y_i = points[i, y]
            z_i = points[i, z]
            inv_r = rsqrt_r(abs2(x_i) + abs2(y_i) + abs2(z_i))
            points[i, x] = x_i * inv_r
            points[i, y] = y_i * inv_r
            points[i, z] = z_i * inv_r
        end
    end
    return points
end


function stereographic_coordinates!(
    result::AbstractMatrix{T},
    points::AbstractMatrix{T},
) where {T}
    point_axis = axes(result, 1)
    st_axis = axes(result, 2)
    @assert point_axis == axes(points, 1)
    xyz_axis = axes(points, 2)
    @assert length(st_axis) == 2
    @assert length(xyz_axis) == 3
    s, t = st_axis
    x, y, z = xyz_axis
    _one = one(T)
    @inbounds begin
        @simd ivdep for i in point_axis
            inv_1z = inv(_one + abs(points[i, z]))
            result[i, s] = points[i, x] * inv_1z
            result[i, t] = points[i, y] * inv_1z
        end
    end
    return result
end


function sphere_step!(
    new_points::AbstractMatrix{T},
    new_stereo_coords::AbstractMatrix{T},
    points::AbstractMatrix{T},
    stereo_coords::AbstractMatrix{T},
    step_size::T,
    step_direction::AbstractVector{T},
) where {T}
    point_axis = axes(new_points, 1)
    xyz_axis = axes(new_points, 2)
    @assert point_axis == axes(new_stereo_coords, 1)
    st_axis = axes(new_stereo_coords, 2)
    @assert point_axis == axes(points, 1)
    @assert xyz_axis == axes(points, 2)
    @assert point_axis == axes(stereo_coords, 1)
    @assert st_axis == axes(stereo_coords, 2)
    reduced_axis = axes(step_direction, 1)
    @assert length(reduced_axis) == 2 * length(point_axis)
    @assert length(xyz_axis) == 3
    @assert length(st_axis) == 2
    x, y, z = xyz_axis
    s, t = st_axis
    _one = one(T)
    @inbounds begin
        @simd ivdep for i in point_axis
            u = first(reduced_axis) + 2 * (i - first(point_axis))
            v = u + 1
            x_i = points[i, x]
            y_i = points[i, y]
            z_i = points[i, z]
            step_u = step_size * step_direction[u]
            step_v = flipsign(step_size * step_direction[v], z_i)
            inv_r = rsqrt_r(_one + abs2(step_u) + abs2(step_v))
            xy_scale = _one - muladd(
                stereo_coords[i, s], step_u, stereo_coords[i, t] * step_v)
            xy_overlap = muladd(x_i, step_u, y_i * step_v)
            x_i = muladd(xy_scale, x_i, step_u) * inv_r
            y_i = muladd(xy_scale, y_i, step_v) * inv_r
            z_i = (z_i - flipsign(xy_overlap, z_i)) * inv_r
            inv_1z = inv(_one + abs(z_i))
            new_points[i, x] = x_i
            new_points[i, y] = y_i
            new_points[i, z] = z_i
            new_stereo_coords[i, s] = x_i * inv_1z
            new_stereo_coords[i, t] = y_i * inv_1z
        end
    end
    return new_points
end


function sphere_tangent!(
    vectors::AbstractMatrix{T},
    points::AbstractMatrix{T},
) where {T}
    point_axis = axes(vectors, 1)
    xyz_axis = axes(vectors, 2)
    @assert point_axis == axes(points, 1)
    @assert xyz_axis == axes(points, 2)
    @assert length(xyz_axis) == 3
    x, y, z = xyz_axis
    result = zero(T)
    @inbounds begin
        @simd ivdep for i in point_axis
            x_i = points[i, x]
            y_i = points[i, y]
            z_i = points[i, z]
            vx_i = vectors[i, x]
            vy_i = vectors[i, y]
            vz_i = vectors[i, z]
            overlap = muladd(x_i, vx_i, muladd(y_i, vy_i, z_i * vz_i))
            vx_i = muladd(-overlap, x_i, vx_i)
            vy_i = muladd(-overlap, y_i, vy_i)
            vz_i = muladd(-overlap, z_i, vz_i)
            vectors[i, x] = vx_i
            vectors[i, y] = vy_i
            vectors[i, z] = vz_i
            result += abs2(vx_i) + abs2(vy_i) + abs2(vz_i)
        end
    end
    return result
end


function sphere_killing_vectors!(
    result::AbstractMatrix{T},
    points::AbstractMatrix{T},
    stereo_coords::AbstractMatrix{T},
) where {T}
    reduced_axis = axes(result, 1)
    rotation_axis = axes(result, 2)
    point_axis = axes(points, 1)
    xyz_axis = axes(points, 2)
    @assert point_axis == axes(stereo_coords, 1)
    st_axis = axes(stereo_coords, 2)
    @assert length(reduced_axis) == 2 * length(point_axis)
    @assert length(rotation_axis) == 3
    @assert length(xyz_axis) == 3
    @assert length(st_axis) == 2
    rx, ry, rz = rotation_axis
    x, y, z = xyz_axis
    s, t = st_axis
    _one = one(T)
    @inbounds begin
        @simd ivdep for i in point_axis
            u = first(reduced_axis) + 2 * (i - first(point_axis))
            v = u + 1
            x_i = points[i, x]
            y_i = points[i, y]
            z_i = points[i, z]
            xs_i = x_i * stereo_coords[i, s]
            xt_i = x_i * stereo_coords[i, t]
            result[u, rx] = -flipsign(xt_i, z_i)
            result[v, rx] = xs_i - _one
            result[u, ry] = flipsign(xs_i + abs(z_i), z_i)
            result[v, ry] = xt_i
            result[u, rz] = -y_i
            result[v, rz] = flipsign(x_i, z_i)
        end
    end
    return result
end


####################################################### SPHERICAL COULOMB ENERGY


export construct_pair_geometry!, hessian_from_pair_geometry!


function construct_pair_geometry!(
    H::AbstractMatrix{T},
    forces_uv::AbstractVector{T},
    points::AbstractMatrix{T},
    stereo_coords::AbstractMatrix{T},
) where {T}
    reduced_axis = axes(H, 1)
    @assert reduced_axis == axes(H, 2)
    @assert reduced_axis == axes(forces_uv, 1)
    point_axis = axes(points, 1)
    xyz_axis = axes(points, 2)
    @assert point_axis == axes(stereo_coords, 1)
    st_axis = axes(stereo_coords, 2)
    @assert length(reduced_axis) == 2 * length(point_axis)
    @assert length(xyz_axis) == 3
    @assert length(st_axis) == 2
    x, y, z = xyz_axis
    s, t = st_axis
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _half = inv(_two)
    result = _zero
    fill!(forces_uv, _zero)
    @inbounds begin
        for i in point_axis
            iu = first(reduced_axis) + 2 * (i - first(point_axis))
            iv = iu + 1
            x_i = points[i, x]
            y_i = points[i, y]
            z_i = points[i, z]
            s_i = stereo_coords[i, s]
            t_i = stereo_coords[i, t]
            fu_i = forces_uv[iu]
            fv_i = forces_uv[iv]
            @simd ivdep for j = (i+1):last(point_axis)
                ju = first(reduced_axis) + 2 * (j - first(point_axis))
                jv = ju + 1
                x_j = points[j, x]
                y_j = points[j, y]
                z_j = points[j, z]
                s_j = stereo_coords[j, s]
                t_j = stereo_coords[j, t]
                dx = x_i - x_j
                dy = y_i - y_j
                dz = z_i - z_j
                r2 = abs2(dx) + abs2(dy) + abs2(dz)
                overlap = muladd(-_half, r2, _one)
                inv_r = rsqrt_r(r2)
                inv_r3 = abs2(inv_r) * inv_r
                overlap_i = overlap + flipsign(z_j, z_i)
                overlap_j = overlap + flipsign(z_i, z_j)
                du_i = muladd(s_i, overlap_i, -x_j)
                du_j = muladd(-s_j, overlap_j, x_i)
                dv_i = flipsign(muladd(t_i, overlap_i, -y_j), z_i)
                dv_j = flipsign(muladd(-t_j, overlap_j, y_i), z_j)
                fu_i = muladd(inv_r3, du_i, fu_i)
                fv_i = muladd(inv_r3, dv_i, fv_i)
                forces_uv[ju] = muladd(-inv_r3, du_j, forces_uv[ju])
                forces_uv[jv] = muladd(-inv_r3, dv_j, forces_uv[jv])
                H[ju, iu] = inv_r
                H[jv, iu] = overlap
                H[ju, iv] = du_i
                H[jv, iv] = dv_i
            end
            result += abs2(fu_i) + abs2(fv_i)
            forces_uv[iu] = fu_i
            forces_uv[iv] = fv_i
        end
    end
    return result
end


function hessian_from_pair_geometry!(
    H::AbstractMatrix{T},
    points::AbstractMatrix{T},
    stereo_coords::AbstractMatrix{T},
) where {T}
    reduced_axis = axes(H, 1)
    @assert reduced_axis == axes(H, 2)
    point_axis = axes(points, 1)
    xyz_axis = axes(points, 2)
    @assert point_axis == axes(stereo_coords, 1)
    st_axis = axes(stereo_coords, 2)
    @assert length(reduced_axis) == 2 * length(point_axis)
    @assert length(xyz_axis) == 3
    @assert length(st_axis) == 2
    x, y, z = xyz_axis
    s, t = st_axis
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _three = _two + _one
    @inbounds begin
        @simd ivdep for i in point_axis
            iu = first(reduced_axis) + 2 * (i - first(point_axis))
            iv = iu + 1
            H[iu, iu] = _zero
            H[iv, iu] = _zero
            H[iv, iv] = _zero
        end
        for i = first(point_axis):(last(point_axis)-1)
            iu = first(reduced_axis) + 2 * (i - first(point_axis))
            iv = iu + 1
            x_i = points[i, x]
            y_i = points[i, y]
            z_i = points[i, z]
            xs_i = x_i * stereo_coords[i, s]
            xt_i = x_i * stereo_coords[i, t]
            ux_i = _one - xs_i
            uy_i = -xt_i
            vx_i = -flipsign(xt_i, z_i)
            vy_i = flipsign(xs_i + abs(z_i), z_i)
            h_uu_i = _zero
            h_vu_i = _zero
            h_vv_i = _zero
            @simd ivdep for j = (i+1):last(point_axis)
                ju = first(reduced_axis) + 2 * (j - first(point_axis))
                jv = ju + 1
                z_j = points[j, z]
                s_j = stereo_coords[j, s]
                t_j = stereo_coords[j, t]
                inv_r = H[ju, iu]
                inv_r2 = abs2(inv_r)
                inv_r3 = inv_r2 * inv_r
                three_inv_r5 = _three * inv_r3 * inv_r2
                overlap = H[jv, iu]
                overlap_j = overlap + flipsign(z_i, z_j)
                du_i = H[ju, iv]
                dv_i = H[jv, iv]
                du_j = muladd(-s_j, overlap_j, x_i)
                dv_j = flipsign(muladd(-t_j, overlap_j, y_i), z_j)
                scaled_du_i = three_inv_r5 * du_i
                scaled_dv_i = three_inv_r5 * dv_i
                scaled_du_j = three_inv_r5 * du_j
                scaled_dv_j = three_inv_r5 * dv_j
                qu = -du_i - flipsign(flipsign(x_i, z_i), z_j)
                qv = -dv_i - flipsign(y_i, z_j)
                q_uu = muladd(-s_j, qu, ux_i)
                q_vu = flipsign(muladd(-t_j, qu, uy_i), z_j)
                q_uv = muladd(-s_j, qv, vx_i)
                q_vv = flipsign(muladd(-t_j, qv, vy_i), z_j)
                H[ju, iu] = muladd(q_uu, inv_r3, -du_j * scaled_du_i)
                H[jv, iu] = muladd(q_vu, inv_r3, -dv_j * scaled_du_i)
                H[ju, iv] = muladd(q_uv, inv_r3, -du_j * scaled_dv_i)
                H[jv, iv] = muladd(q_vv, inv_r3, -dv_j * scaled_dv_i)
                diag_shift = -overlap * inv_r3
                h_uu_i = muladd(du_i, scaled_du_i, h_uu_i + diag_shift)
                h_vu_i = muladd(dv_i, scaled_du_i, h_vu_i)
                h_vv_i = muladd(dv_i, scaled_dv_i, h_vv_i + diag_shift)
                H[ju, ju] = muladd(du_j, scaled_du_j, H[ju, ju] + diag_shift)
                H[jv, ju] = muladd(dv_j, scaled_du_j, H[jv, ju])
                H[jv, jv] = muladd(dv_j, scaled_dv_j, H[jv, jv] + diag_shift)
            end
            H[iu, iu] += h_uu_i
            H[iv, iu] += h_vu_i
            H[iv, iv] += h_vv_i
        end
    end
    return H
end


################################################################################

end # module ThomSolve
