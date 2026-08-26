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
    @inbounds for i = first(point_axis):(last(point_axis)-1)
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
            r2 = abs2(dx) + abs2(dy) + abs2(dz)
            result += rsqrt_r(r2)
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
                r2 = abs2(dx) + abs2(dy) + abs2(dz)
                inv_r = rsqrt_r(r2)
                inv_r2 = abs2(inv_r)
                point_energy_i += inv_r
                point_energies[j] += inv_r
                inv_r3 = inv_r2 * inv_r
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


export sphere_project!, sphere_step!, sphere_tangent!,
    stereographic_coordinates!, sphere_killing_vectors!


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
            r2 = abs2(x_i) + abs2(y_i) + abs2(z_i)
            inv_r = rsqrt_r(r2)
            points[i, x] = x_i * inv_r
            points[i, y] = y_i * inv_r
            points[i, z] = z_i * inv_r
        end
    end
    return points
end


function sphere_step!(
    result::AbstractMatrix{T},
    points::AbstractMatrix{T},
    step_size::T,
    step_direction::AbstractMatrix{T},
) where {T}
    point_axis = axes(result, 1)
    xyz_axis = axes(result, 2)
    @assert point_axis == axes(points, 1)
    @assert xyz_axis == axes(points, 2)
    @assert point_axis == axes(step_direction, 1)
    @assert xyz_axis == axes(step_direction, 2)
    @assert length(xyz_axis) == 3
    x, y, z = xyz_axis
    @inbounds begin
        @simd ivdep for i in point_axis
            x_i = muladd(step_size, step_direction[i, x], points[i, x])
            y_i = muladd(step_size, step_direction[i, y], points[i, y])
            z_i = muladd(step_size, step_direction[i, z], points[i, z])
            r2 = abs2(x_i) + abs2(y_i) + abs2(z_i)
            inv_r = rsqrt_r(r2)
            result[i, x] = x_i * inv_r
            result[i, y] = y_i * inv_r
            result[i, z] = z_i * inv_r
        end
    end
    return result
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


function stereographic_coordinates!(
    result::AbstractMatrix{T},
    points::AbstractMatrix{T},
) where {T}
    point_axis = axes(result, 1)
    stereo_axis = axes(result, 2)
    @assert point_axis == axes(points, 1)
    xyz_axis = axes(points, 2)
    @assert length(stereo_axis) == 2
    @assert length(xyz_axis) == 3
    s, t = stereo_axis
    x, y, z = xyz_axis
    _one = one(T)
    @inbounds begin
        @simd ivdep for i in point_axis
            inv_z1 = inv(_one + abs(points[i, z]))
            result[i, s] = points[i, x] * inv_z1
            result[i, t] = points[i, y] * inv_z1
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


################################################################################

end # module ThomSolve
