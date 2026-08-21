module ThomSolve


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
            result += sqrt(inv(r2))
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
                inv_r2 = inv(r2)
                inv_r = sqrt(inv_r2)
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


################################################################ SPHERE GEOMETRY


export sphere_project!, sphere_tangent!


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
            inv_r = sqrt(inv(r2))
            points[i, x] = inv_r * x_i
            points[i, y] = inv_r * y_i
            points[i, z] = inv_r * z_i
        end
    end
    return points
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


end # module ThomSolve
