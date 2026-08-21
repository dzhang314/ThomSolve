module LBL


export LBLPivotType, lbl_factorize!, lbl_solve!, lbl_inertia


function swap_symmetric!(A::AbstractMatrix, i::Int, j::Int)
    # A is assumed to be symmetric; only its lower triangle (i >= j) is used.
    indices = axes(A, 1)
    @assert indices == axes(A, 2)
    @assert i in indices
    @assert j in indices
    if i == j
        return A
    end
    i, j = minmax(i, j)
    @inbounds begin
        @simd ivdep for k = first(indices):(i-1)
            A[i, k], A[j, k] = A[j, k], A[i, k]
        end
        A[i, i], A[j, j] = A[j, j], A[i, i]
        @simd ivdep for k = (i+1):(j-1)
            A[k, i], A[j, k] = A[j, k], A[k, i]
        end
        @simd ivdep for k = (j+1):last(indices)
            A[k, i], A[k, j] = A[k, j], A[k, i]
        end
    end
    return A
end


@enum LBLPivotType::UInt8 LBL_PIVOT_CONT LBL_PIVOT_1X1 LBL_PIVOT_2X2


function lbl_factorize!(
    A::AbstractMatrix{T},
    permutation::AbstractVector{Int},
    pivot_type::AbstractVector{LBLPivotType},
) where {T}
    # A is assumed to be symmetric; only its lower triangle (i >= j) is used.
    indices = axes(A, 1)
    @assert indices == axes(A, 2)
    @assert indices == axes(permutation, 1)
    @assert indices == axes(pivot_type, 1)
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _four = _two + _two
    _eight = _four + _four
    _sixteen = _eight + _eight
    _seventeen = _sixteen + _one
    _alpha = (_one + sqrt(_seventeen)) / _eight
    @inbounds begin
        permutation .= indices
        k = first(indices)
        while k < last(indices)
            max_abs_col = _zero
            p = k + 1
            for i = (k+1):last(indices)
                abs_col = abs(A[i, k])
                if abs_col > max_abs_col
                    max_abs_col = abs_col
                    p = i
                end
            end
            abs_diag = abs(A[k, k])
            if abs_diag >= _alpha * max_abs_col
                pivot_type[k] = LBL_PIVOT_1X1
            else
                max_abs_row = _zero
                for i = k:last(indices)
                    if i != p
                        abs_row = abs((i < p) ? A[p, i] : A[i, p])
                        if abs_row > max_abs_row
                            max_abs_row = abs_row
                        end
                    end
                end
                if abs_diag * max_abs_row >= _alpha * abs2(max_abs_col)
                    pivot_type[k] = LBL_PIVOT_1X1
                elseif abs(A[p, p]) >= _alpha * max_abs_row
                    swap_symmetric!(A, k, p)
                    permutation[k], permutation[p] =
                        permutation[p], permutation[k]
                    pivot_type[k] = LBL_PIVOT_1X1
                else
                    swap_symmetric!(A, k + 1, p)
                    permutation[k+1], permutation[p] =
                        permutation[p], permutation[k+1]
                    pivot_type[k] = LBL_PIVOT_2X2
                    pivot_type[k+1] = LBL_PIVOT_CONT
                end
            end
            if pivot_type[k] == LBL_PIVOT_1X1
                pivot = A[k, k]
                if iszero(pivot)
                    A[k, last(indices)] = _zero
                    @simd ivdep for i = (k+1):last(indices)
                        A[i, k] = _zero
                    end
                else
                    inv_pivot = inv(pivot)
                    A[k, last(indices)] = inv_pivot
                    @simd ivdep for i = (k+1):last(indices)
                        A[i, k] *= inv_pivot
                    end
                    for j = (k+1):last(indices)
                        schur_jk = -pivot * A[j, k]
                        @simd ivdep for i = j:last(indices)
                            A[i, j] = muladd(A[i, k], schur_jk, A[i, j])
                        end
                    end
                end
                k += 1
            else
                p_11 = A[k, k]
                p_12 = A[k+1, k]
                p_22 = A[k+1, k+1]
                det_pivot = muladd(p_11, p_22, -abs2(p_12))
                if iszero(det_pivot)
                    A[k, last(indices)] = _zero
                    @simd ivdep for i = (k+2):last(indices)
                        A[i, k] = _zero
                        A[i, k+1] = _zero
                    end
                else
                    inv_det_pivot = inv(det_pivot)
                    A[k, last(indices)] = inv_det_pivot
                    inv_p_11 = inv_det_pivot * p_22
                    inv_p_12 = -inv_det_pivot * p_12
                    inv_p_22 = inv_det_pivot * p_11
                    @simd ivdep for i = (k+2):last(indices)
                        a_i1 = A[i, k]
                        a_i2 = A[i, k+1]
                        A[i, k] = muladd(inv_p_11, a_i1, inv_p_12 * a_i2)
                        A[i, k+1] = muladd(inv_p_12, a_i1, inv_p_22 * a_i2)
                    end
                    for j = (k+2):last(indices)
                        a_j1 = A[j, k]
                        a_j2 = A[j, k+1]
                        schur_j1 = -muladd(p_11, a_j1, p_12 * a_j2)
                        schur_j2 = -muladd(p_12, a_j1, p_22 * a_j2)
                        @simd ivdep for i = j:last(indices)
                            A[i, j] = muladd(A[i, k], schur_j1,
                                muladd(A[i, k+1], schur_j2, A[i, j]))
                        end
                    end
                end
                k += 2
            end
        end
        if k == last(indices)
            pivot_type[k] = LBL_PIVOT_1X1
        end
    end
    return A
end


function lbl_solve!(
    x::AbstractVector{T},
    x_perm::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    permutation::AbstractVector{Int},
    pivot_type::AbstractVector{LBLPivotType},
) where {T<:AbstractFloat}
    indices = axes(x, 1)
    @assert indices == axes(x_perm, 1)
    @assert indices == axes(A, 1)
    @assert indices == axes(A, 2)
    @assert indices == axes(b, 1)
    @assert indices == axes(permutation, 1)
    @assert indices == axes(pivot_type, 1)
    @inbounds begin
        @simd ivdep for i in indices
            x_perm[i] = b[permutation[i]]
        end
        for i in indices
            rhs_i = x_perm[i]
            for j = first(indices):(i-2)
                rhs_i = muladd(-A[i, j], x_perm[j], rhs_i)
            end
            if (i > first(indices)) && (pivot_type[i-1] != LBL_PIVOT_2X2)
                rhs_i = muladd(-A[i, i-1], x_perm[i-1], rhs_i)
            end
            x_perm[i] = rhs_i
        end
        k = first(indices)
        while k <= last(indices)
            if pivot_type[k] == LBL_PIVOT_1X1
                if k < last(indices)
                    x_perm[k] *= A[k, last(indices)]
                else
                    pivot = A[k, k]
                    x_perm[k] = iszero(pivot) ? zero(T) : x_perm[k] / pivot
                end
                k += 1
            else
                p_11 = A[k, k]
                p_12 = A[k+1, k]
                p_22 = A[k+1, k+1]
                inv_det_pivot = A[k, last(indices)]
                rhs_1 = x_perm[k]
                rhs_2 = x_perm[k+1]
                x_perm[k] = inv_det_pivot * muladd(p_22, rhs_1, -p_12 * rhs_2)
                x_perm[k+1] = inv_det_pivot * muladd(p_11, rhs_2, -p_12 * rhs_1)
                k += 2
            end
        end
        for i = last(indices):-1:first(indices)
            rhs_i = x_perm[i]
            @simd for j = (i+2):last(indices)
                rhs_i = muladd(-A[j, i], x_perm[j], rhs_i)
            end
            if (i < last(indices)) && (pivot_type[i] != LBL_PIVOT_2X2)
                rhs_i = muladd(-A[i+1, i], x_perm[i+1], rhs_i)
            end
            x_perm[i] = rhs_i
        end
        @simd ivdep for i in indices
            x[permutation[i]] = x_perm[i]
        end
    end
    return x
end


function lbl_inertia(
    A::AbstractMatrix{T},
    pivot_type::AbstractVector{LBLPivotType},
) where {T}
    indices = axes(A, 1)
    @assert indices == axes(A, 2)
    @assert indices == axes(pivot_type, 1)
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _four = _two + _two
    _half = inv(_two)
    negative_count = 0
    zero_count = 0
    positive_count = 0
    @inbounds begin
        max_abs_diag = _zero
        for i in indices
            abs_diag = abs(A[i, i])
            if abs_diag > max_abs_diag
                max_abs_diag = abs_diag
            end
        end
        zero_tol = length(indices) * eps(T) * max(_one, max_abs_diag)
        k = first(indices)
        @inbounds while k <= last(indices)
            if pivot_type[k] == LBL_PIVOT_1X1
                pivot = A[k, k]
                if abs(pivot) <= zero_tol
                    zero_count += 1
                elseif signbit(pivot)
                    negative_count += 1
                else
                    positive_count += 1
                end
                k += 1
            else
                p_11 = A[k, k]
                p_12 = A[k+1, k]
                p_22 = A[k+1, k+1]
                trace = p_11 + p_22
                det_pivot = muladd(p_11, p_22, -abs2(p_12))
                discriminant = sqrt(max(_zero,
                    muladd(-_four, det_pivot, abs2(trace))))
                lambda_1 = _half * (trace + discriminant)
                lambda_2 = _half * (trace - discriminant)
                if abs(lambda_1) <= zero_tol
                    zero_count += 1
                elseif signbit(lambda_1)
                    negative_count += 1
                else
                    positive_count += 1
                end
                if abs(lambda_2) <= zero_tol
                    zero_count += 1
                elseif signbit(lambda_2)
                    negative_count += 1
                else
                    positive_count += 1
                end
                k += 2
            end
        end
    end
    return (negative_count, zero_count, positive_count)
end


end # module LBL
