import math
import sys
import os
try:
    from PIL import Image
except Exception:
    Image = None

# --- Analytical Models for Resource Consumption ---

# Energy constants (example values in Joules per operation)
# These are illustrative and would be hardware-specific in a real scenario.
ENERGY_PER_MAC_FP32 = 3.7e-12  # Energy for a 32-bit floating-point MAC operation
ENERGY_PER_ADD_FP32 = 0.9e-12  # Energy for a 32-bit floating-point ADD operation
ENERGY_PER_MAC_INT8 = 0.6e-12  # Approx: INT8 MACs are cheaper
ENERGY_PER_ADD_INT8 = 0.2e-12  # Approx: INT8 adds are cheaper
ENERGY_PER_BYTE_ACCESS = 5.0e-12 # Energy to access one byte from memory (e.g., DRAM)

# --- Helper Functions for CNN Layers ---

def conv2d(input_map, kernel, stride=1, padding=0, bias=0.0):
    """
    Performs a 2D convolution for a single channel.

    Args:
        input_map (list of lists): Input feature map (H x W).
        kernel (list of lists): Convolution kernel (K x K).
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.

    Returns:
        tuple: (output_map, flops, memory_bytes, energy_j)
    """
    # Get dimensions
    H_in = len(input_map)
    W_in = len(input_map[0])
    # Support codebook-based kernels: {"codebook": [...], "indices": 2D list}
    is_cb = isinstance(kernel, dict) and ('codebook' in kernel) and ('indices' in kernel)
    K = len(kernel['indices']) if is_cb else len(kernel)
    
    # Apply padding (simple zero-padding)
    if padding > 0:
        padded_input = [[0] * (W_in + 2 * padding) for _ in range(H_in + 2 * padding)]
        for i in range(H_in):
            for j in range(W_in):
                padded_input[i + padding][j + padding] = input_map[i][j]
        input_map = padded_input
        H_in += 2 * padding
        W_in += 2 * padding

    # Calculate output dimensions
    H_out = (H_in - K) // stride + 1
    W_out = (W_in - K) // stride + 1
    
    output_map = [[0] * W_out for _ in range(H_out)]

    # Precompute non-zero kernel entries for sparsity-aware compute
    nz_positions = []
    if is_cb:
        cb = kernel['codebook']
        idxs = kernel['indices']
        for ki in range(K):
            for kj in range(K):
                kv = cb[idxs[ki][kj]]
                if kv != 0:
                    nz_positions.append((ki, kj, kv))
    else:
        for ki in range(K):
            for kj in range(K):
                if kernel[ki][kj] != 0:
                    nz_positions.append((ki, kj, kernel[ki][kj]))
    
    flops = 0
    
    # Perform convolution (sparsity-aware)
    for i in range(H_out):
        for j in range(W_out):
            # Extract the receptive field
            receptive_field = []
            for row_idx in range(i * stride, i * stride + K):
                receptive_field.append(input_map[row_idx][j * stride:j * stride + K])
            
            # Accumulate only on non-zero kernel entries
            conv_sum = 0
            for ki, kj, kv in nz_positions:
                conv_sum += receptive_field[ki][kj] * kv
                flops += 2 # 1 mul + 1 add
            # Add bias
            conv_sum += bias
            flops += 1  # bias add
            output_map[i][j] = conv_sum

    # --- Resource Calculation ---
    # FLOPs: Already calculated
    
    # Memory: Input size + Kernel size + Output size (in bytes)
    input_memory = H_in * W_in * 4  # FP32 input
    if is_cb:
        # Codebook (FP32) + indices (1/2/4 bytes)
        Kc = len(kernel['codebook'])
        # choose index bytes based on Kc
        if Kc <= 256:
            idx_bytes = 1
        elif Kc <= 65536:
            idx_bytes = 2
        else:
            idx_bytes = 4
        kernel_memory = (Kc * 4) + (K * K * idx_bytes)
    else:
        kernel_memory = K * K * 4
    output_memory = H_out * W_out * 4
    total_memory_bytes = input_memory + kernel_memory + output_memory
    
    # Energy: Based on MAC operations and memory access
    num_macs = H_out * W_out * len(nz_positions)
    compute_energy = num_macs * ENERGY_PER_MAC_FP32
    memory_energy = total_memory_bytes * ENERGY_PER_BYTE_ACCESS
    total_energy_j = compute_energy + memory_energy

    return output_map, flops, total_memory_bytes, total_energy_j

def relu(input_map):
    """
    Applies the ReLU activation function element-wise.
    """
    H = len(input_map)
    W = len(input_map[0])
    output_map = [[max(0, val) for val in row] for row in input_map]
    
    # Resource Calculation
    flops = H * W  # One comparison per element
    memory_bytes = H * W * 4 * 2 # Input + Output
    energy_j = (flops * ENERGY_PER_ADD_FP32) + (memory_bytes * ENERGY_PER_BYTE_ACCESS)
    
    return output_map, flops, memory_bytes, energy_j

def avg_pool2d(input_map, pool_size, stride):
    """Performs 2D average pooling."""
    H_in, W_in = len(input_map), len(input_map[0])
    H_out = (H_in - pool_size) // stride + 1
    W_out = (W_in - pool_size) // stride + 1

    output_map = [[0] * W_out for _ in range(H_out)]
    flops = 0
    for i in range(H_out):
        for j in range(W_out):
            s = 0.0
            for row_idx in range(i * stride, i * stride + pool_size):
                row = input_map[row_idx][j * stride:j * stride + pool_size]
                for v in row:
                    s += v
                    flops += 1  # add
            # divide by count
            count = pool_size * pool_size
            output_map[i][j] = s / count
            flops += 1  # division
    memory_bytes = (H_in * W_in * 4) + (H_out * W_out * 4)
    energy_j = (flops * ENERGY_PER_ADD_FP32) + (memory_bytes * ENERGY_PER_BYTE_ACCESS)
    return output_map, flops, memory_bytes, energy_j

def conv2d_multi(input_tensor, kernels, stride=1, padding=0, bias=0.0):
    """
    Multi-channel convolution for one output map.
    input_tensor: list[channels][H][W]
    kernels: list[channels][K][K] (one kernel per input channel)
    bias: scalar bias added to each output element

    Returns: (output_map, flops, memory_bytes, energy_j)
    """
    C = len(input_tensor)
    # Convolve per channel and accumulate
    per_ch_outputs = []
    total_flops = 0
    total_macs = 0
    # We'll compute dimensions from first channel
    for c in range(C):
        out_c, flops_c, mem_c, energy_c = conv2d(input_tensor[c], kernels[c], stride=stride, padding=padding, bias=0.0)
        per_ch_outputs.append(out_c)
        total_flops += flops_c
        # estimate macs from flops_c/2 approximately (mul+add per MAC), minus bias adds (we set bias=0.0)
        # Here we recompute exact macs as H_out*W_out*nnz for this channel
        K = len(kernels[c])
        # count non-zeros
        nnz = 0
        for i in range(K):
            for j in range(K):
                if kernels[c][i][j] != 0:
                    nnz += 1
        H_out = len(out_c)
        W_out = len(out_c[0]) if H_out > 0 else 0
        total_macs += H_out * W_out * nnz
    # Sum across channels and add bias
    H_out = len(per_ch_outputs[0])
    W_out = len(per_ch_outputs[0][0]) if H_out > 0 else 0
    output = [[0.0] * W_out for _ in range(H_out)]
    for i in range(H_out):
        for j in range(W_out):
            acc = 0.0
            for c in range(C):
                acc += per_ch_outputs[c][i][j]
                total_flops += 1  # add for channel accumulation
            acc += bias
            total_flops += 1  # bias add
            output[i][j] = acc
    # Memory approximation
    H_in = len(input_tensor[0])
    W_in = len(input_tensor[0][0])
    input_memory = C * H_in * W_in * 4
    # Support codebook-form kernels for memory accounting
    if isinstance(kernels[0], dict) and ('codebook' in kernels[0]) and ('indices' in kernels[0]):
        K = len(kernels[0]['indices'])
        kernel_memory = 0
        for kc in kernels:
            Kc = len(kc['codebook'])
            if Kc <= 256:
                idx_bytes = 1
            elif Kc <= 65536:
                idx_bytes = 2
            else:
                idx_bytes = 4
            kernel_memory += (Kc * 4) + (K * K * idx_bytes)
    else:
        K = len(kernels[0])
        kernel_memory = C * K * K * 4
    output_memory = H_out * W_out * 4
    memory_bytes = input_memory + kernel_memory + output_memory
    compute_energy = total_macs * ENERGY_PER_MAC_FP32
    energy_j = compute_energy + (memory_bytes * ENERGY_PER_BYTE_ACCESS)
    return output, total_flops, memory_bytes, energy_j

# ---------------- Energy-efficient pure-math paths -----------------

def conv2d_int8(input_map, kernel, stride=1, padding=0, bias=0.0):
    """
    INT8 execution path: quantize input and kernel to int8, perform int32 accumulation,
    then dequantize output. Energy model uses INT8 constants.
    Returns (output_map_float, flops, memory_bytes, energy_j)
    """
    # Quantize
    q_in, s_in, _ = quantize_tensor(input_map, bits=8)
    q_k, s_k, _ = quantize_tensor(kernel, bits=8)
    # Use same conv shape computation as conv2d
    H_in, W_in = len(q_in), len(q_in[0])
    K = len(q_k)
    H_pad = H_in + 2 * padding
    W_pad = W_in + 2 * padding
    padded = [[0] * W_pad for _ in range(H_pad)]
    for i in range(H_in):
        for j in range(W_in):
            padded[i + padding][j + padding] = q_in[i][j]
    H_out = (H_pad - K) // stride + 1
    W_out = (W_pad - K) // stride + 1
    output_q = [[0] * W_out for _ in range(H_out)]
    # Precompute non-zero kernel entries
    nz = []
    for ki in range(K):
        for kj in range(K):
            kv = q_k[ki][kj]
            if kv != 0:
                nz.append((ki, kj, kv))
    flops = 0
    for i in range(H_out):
        for j in range(W_out):
            acc = 0
            # If receptive field is all zeros, skip work
            zero_block = True
            for ki, kj, kv in nz:
                v = padded[i * stride + ki][j * stride + kj]
                if v != 0:
                    zero_block = False
                    acc += v * kv
                    flops += 1  # approximate: count MAC as 1
            # bias in float domain -> scale to int8 scale domain approx
            acc_f = acc * (s_in * s_k) + bias
            output_q[i][j] = acc_f
            if zero_block:
                # minimal cost for checking zeros (already accounted implicitly)
                pass
    # Memory model (INT8 in/out, but we dequantize to float map for return)
    input_memory = H_in * W_in  # bytes (int8)
    kernel_memory = K * K       # bytes (int8)
    output_memory = H_out * W_out * 4  # we return float
    memory_bytes = input_memory + kernel_memory + output_memory
    # Energy
    compute_energy = flops * ENERGY_PER_MAC_INT8
    energy_j = compute_energy + memory_bytes * ENERGY_PER_BYTE_ACCESS
    # Convert to float map already in output_q as floats
    return output_q, flops, memory_bytes, energy_j

def conv1d_row(input_map, kernel1d, stride=1, padding=0):
    H, W = len(input_map), len(input_map[0])
    K = len(kernel1d)
    W_pad = W + 2 * padding
    padded = [[0] * W_pad for _ in range(H)]
    for i in range(H):
        for j in range(W):
            padded[i][j + padding] = input_map[i][j]
    W_out = (W_pad - K) // stride + 1
    out = [[0.0] * W_out for _ in range(H)]
    flops = 0
    for i in range(H):
        for j in range(W_out):
            s = 0.0
            for t in range(K):
                s += padded[i][j * stride + t] * kernel1d[t]
                flops += 2
            out[i][j] = s
    mem = (H * W * 4) + (K * 4) + (H * W_out * 4)
    energy = flops * ENERGY_PER_MAC_FP32 + mem * ENERGY_PER_BYTE_ACCESS
    return out, flops, mem, energy

def conv1d_col(input_map, kernel1d, stride=1, padding=0):
    H, W = len(input_map), len(input_map[0])
    K = len(kernel1d)
    H_pad = H + 2 * padding
    padded = [[0] * W for _ in range(H_pad)]
    for i in range(H):
        for j in range(W):
            padded[i + padding][j] = input_map[i][j]
    H_out = (H_pad - K) // stride + 1
    out = [[0.0] * W for _ in range(H_out)]
    flops = 0
    for i in range(H_out):
        for j in range(W):
            s = 0.0
            for t in range(K):
                s += padded[i * stride + t][j] * kernel1d[t]
                flops += 2
            out[i][j] = s
    mem = (H * W * 4) + (K * 4) + (H_out * W * 4)
    energy = flops * ENERGY_PER_MAC_FP32 + mem * ENERGY_PER_BYTE_ACCESS
    return out, flops, mem, energy

def conv2d_separable(input_map, col_vec, row_vec, stride=1, padding=0, bias=0.0):
    """Apply separable 2D conv by column 1D then row 1D (rank-1 kernel)."""
    tmp, f1, m1, e1 = conv1d_col(input_map, col_vec, stride=1, padding=padding)
    out, f2, m2, e2 = conv1d_row(tmp, row_vec, stride=stride, padding=0)
    # add bias
    H, W = len(out), len(out[0])
    for i in range(H):
        for j in range(W):
            out[i][j] += bias
    flops = f1 + f2 + H * W  # bias adds
    mem = m1 + m2 + (H * W * 4)
    energy = (f1 + f2) * ENERGY_PER_MAC_FP32 + (H * W) * ENERGY_PER_ADD_FP32 + mem * ENERGY_PER_BYTE_ACCESS
    return out, flops, mem, energy

def depthwise_separable_conv(input_tensor, depthwise_kernels, pointwise_weights, stride=1, padding=0, bias=0.0):
    """
    Depthwise separable conv: per-channel KxK conv (depthwise) then 1x1 pointwise combine across channels.
    input_tensor: list[C][H][W]
    depthwise_kernels: list[C][K][K]
    pointwise_weights: list[C] weights for 1x1 combining
    Returns (output_map, flops, memory_bytes, energy_j)
    """
    C = len(input_tensor)
    per_ch = []
    total_flops = 0
    total_macs = 0
    for c in range(C):
        out_c, fl_c, mem_c, en_c = conv2d(input_tensor[c], depthwise_kernels[c], stride=stride, padding=padding, bias=0.0)
        per_ch.append(out_c)
        total_flops += fl_c
        K = len(depthwise_kernels[c])
        # nnz count
        nnz = 0
        for i in range(K):
            for j in range(K):
                if depthwise_kernels[c][i][j] != 0:
                    nnz += 1
        H_out = len(out_c)
        W_out = len(out_c[0]) if H_out > 0 else 0
        total_macs += H_out * W_out * nnz
    # 1x1 combine
    H_out = len(per_ch[0])
    W_out = len(per_ch[0][0]) if H_out > 0 else 0
    out = [[0.0] * W_out for _ in range(H_out)]
    for i in range(H_out):
        for j in range(W_out):
            acc = 0.0
            for c in range(C):
                acc += per_ch[c][i][j] * pointwise_weights[c]
                total_flops += 2  # mul+add
            acc += bias
            total_flops += 1
            out[i][j] = acc
    # Memory
    H_in = len(input_tensor[0])
    W_in = len(input_tensor[0][0])
    K = len(depthwise_kernels[0])
    memory_bytes = (C * H_in * W_in * 4) + (C * K * K * 4) + (C * 4) + (H_out * W_out * 4)
    compute_energy = total_macs * ENERGY_PER_MAC_FP32
    energy_j = compute_energy + memory_bytes * ENERGY_PER_BYTE_ACCESS
    return out, total_flops, memory_bytes, energy_j

# ----------------- Simple image loaders and CLI --------------------

def load_pgm_p2(path):
    """Load ASCII PGM (P2) grayscale into list[list[float]] normalized to [0,1]."""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#')]
    assert lines[0] == 'P2', 'Only P2 PGM supported'
    dims = lines[1].split()
    if len(dims) != 2:
        # Some files may have width/height on next line
        dims += lines[2].split()
        maxval_idx = 3
    else:
        maxval_idx = 2
    w, h = int(dims[0]), int(dims[1])
    maxval = int(lines[maxval_idx])
    data = []
    vals = []
    for tok in ' '.join(lines[maxval_idx+1:]).split():
        vals.append(int(tok))
    assert len(vals) >= w*h
    idx = 0
    for i in range(h):
        row = []
        for j in range(w):
            row.append(vals[idx] / maxval)
            idx += 1
        data.append(row)
    return data

def load_csv_image(path):
    """Load CSV of numbers into list[list[float]]."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = [float(x) for x in line.split(',')]
            data.append(row)
    return data

def load_image(path):
    """Load JPG/PNG image into list[list[float]] normalized to [0,1]."""
    if Image is None:
        raise ImportError("Pillow (PIL) is not installed. Please install with `pip install pillow` to load images.")
    img = Image.open(path).convert('L')  # Convert to grayscale
    w, h = img.size
    pixels = list(img.getdata())
    # Normalize to [0,1] and reshape to 2D
    return [[pixels[i * w + j] / 255.0 for j in range(w)] for i in range(h)]

def edge_detect(img2d, save_path='edge_output.png'):
    """
    Simple Sobel edge detection using conv2d and magnitude combine.
    Returns (mag, flops, memory_bytes, energy_j)
    Saves the edge map as 'edge_output.png' if save_path is provided.
    """
    gx = [[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]
    gy = [[-1,-2,-1],[ 0, 0, 0],[ 1, 2, 1]]
    
    # Apply Sobel filters
    outx, fx, mx, ex = conv2d(img2d, gx, stride=1, padding=1)
    outy, fy, my, ey = conv2d(img2d, gy, stride=1, padding=1)
    
    H, W = len(outx), len(outx[0])
    mag = [[0.0]*W for _ in range(H)]
    flops = fx + fy
    max_val = 0.0
    
    # Compute magnitude and find max for normalization
    for i in range(H):
        for j in range(W):
            v = math.sqrt(outx[i][j]*outx[i][j] + outy[i][j]*outy[i][j])
            mag[i][j] = v
            if v > max_val:
                max_val = v
            flops += 3  # mul+mul+sqrt
    
    # Normalize to [0, 255] and convert to uint8 for saving
    if max_val > 0:
        img_array = []
        for row in mag:
            img_array.append([int(255.0 * (v / max_val)) for v in row])
    else:
        img_array = [[0] * W for _ in range(H)]
    
    # Save the edge map (if PIL available)
    if Image is not None:
        output_img = Image.new('L', (W, H))
        for i in range(H):
            for j in range(W):
                output_img.putpixel((j, i), img_array[i][j])
        output_img.save(save_path)
        print(f"Saved edge map to {save_path}")
    else:
        print("Pillow not installed; skipping image save. Install with `pip install pillow` to enable saving.")
    
    # Calculate memory and energy
    mem = (H*W*4)*3  # Storing input, Gx, Gy, and magnitude maps
    energy = flops * ENERGY_PER_ADD_FP32 + mem * ENERGY_PER_BYTE_ACCESS
    
    return mag, flops, mem, energy

def run_cli(argv):
    if len(argv) >= 3 and argv[1] == 'edge':
        path = argv[2]
        if not os.path.exists(path):
            print(f"Path not found: {path}")
            return
        if path.lower().endswith('.pgm'):
            img = load_pgm_p2(path)
        elif path.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = load_image(path)
        elif path.lower().endswith('.csv'):
            img = load_csv_image(path)
        else:
            print("Unsupported format. Supported: PGM (P2), JPG, PNG, CSV")
            return
        print("Loaded image of size:", len(img), 'x', len(img[0]))
        edges, fl, mem, en = edge_detect(img)
        print("Edge map preview (first 8x8):")
        for i in range(min(8, len(edges))):
            print([f"{v:0.2f}" for v in edges[i][:min(8, len(edges[0]))]])
        print(f"  - FLOPs: {fl}")
        print(f"  - Memory: {mem} bytes")
        print(f"  - Energy: {en:.2e} J")
        
        # Try to open the saved image with default viewer
        try:
            output_path = os.path.join(os.path.dirname(path), 'edge_output.png')
            if os.path.exists(output_path):
                import platform
                if platform.system() == 'Darwin':  # macOS
                    os.system(f'open "{output_path}"')
                elif platform.system() == 'Windows':
                    os.startfile(output_path)
                else:  # Linux
                    os.system(f'xdg-open "{output_path}"')
        except Exception as e:
            print(f"Couldn't open image viewer: {e}")
            print(f"Edge map saved to: {output_path}")
    else:
        # Default to toy demo
        main()

def max_pool2d(input_map, pool_size, stride):
    """
    Performs 2D max pooling.
    """
    H_in, W_in = len(input_map), len(input_map[0])
    H_out = (H_in - pool_size) // stride + 1
    W_out = (W_in - pool_size) // stride + 1
    
    output_map = [[0] * W_out for _ in range(H_out)]
    
    flops = 0
    for i in range(H_out):
        for j in range(W_out):
            pool_window = []
            for row_idx in range(i * stride, i * stride + pool_size):
                pool_window.append(input_map[row_idx][j * stride:j * stride + pool_size])
            
            max_val = -float('inf')
            for row in pool_window:
                for val in row:
                    if val > max_val:
                        max_val = val
            flops += (pool_size * pool_size - 1) # Comparisons to find max
            output_map[i][j] = max_val
            
    # Resource Calculation
    memory_bytes = (H_in * W_in * 4) + (H_out * W_out * 4)
    energy_j = (flops * ENERGY_PER_ADD_FP32) + (memory_bytes * ENERGY_PER_BYTE_ACCESS)

    return output_map, flops, memory_bytes, energy_j

def flatten(input_map):
    """Flatten a 2D list into a 1D list with resource tracking."""
    H, W = len(input_map), len(input_map[0])
    flat = []
    for i in range(H):
        for j in range(W):
            flat.append(input_map[i][j])
    # FLOPs: none, Memory: read + write
    memory_bytes = (H * W * 4) + (H * W * 4)
    energy_j = memory_bytes * ENERGY_PER_BYTE_ACCESS
    return flat, 0, memory_bytes, energy_j

def fully_connected(x_vec, weights, biases=None):
    """
    Simple fully connected layer: y = W x + b
    Args:
        x_vec: list length N_in
        weights: list of lists shape (N_out x N_in)
        biases: list length N_out or None
    Returns:
        (y_vec, flops, memory_bytes, energy_j)
    """
    # Support clustered/codebook weights: {"codebook": list[float], "indices": 2D list N_out x N_in}
    is_cb = isinstance(weights, dict) and ('codebook' in weights) and ('indices' in weights)
    if is_cb:
        N_out = len(weights['indices'])
        N_in = len(weights['indices'][0]) if N_out > 0 else 0
        codebook = weights['codebook']
        idxs2d = weights['indices']
    else:
        N_out = len(weights)
        N_in = len(weights[0]) if N_out > 0 else 0
    y = [0] * N_out
    flops = 0
    for o in range(N_out):
        acc = 0.0
        if is_cb:
            for i in range(N_in):
                w = codebook[idxs2d[o][i]]
                acc += w * x_vec[i]
                flops += 2
        else:
            for i in range(N_in):
                acc += weights[o][i] * x_vec[i]
                flops += 2
        if biases is not None:
            acc += biases[o]
            flops += 1
        y[o] = acc
    # Memory model: input + weights + biases + output (bytes)
    input_mem = N_in * 4
    if is_cb:
        Kc = len(codebook)
        if Kc <= 256:
            idx_bytes = 1
        elif Kc <= 65536:
            idx_bytes = 2
        else:
            idx_bytes = 4
        weight_mem = (Kc * 4) + (N_out * N_in * idx_bytes)
    else:
        weight_mem = N_out * N_in * 4
    bias_mem = (N_out * 4) if biases is not None else 0
    output_mem = N_out * 4
    memory_bytes = input_mem + weight_mem + bias_mem + output_mem
    num_macs = N_out * N_in
    compute_energy = num_macs * ENERGY_PER_MAC_FP32
    memory_energy = memory_bytes * ENERGY_PER_BYTE_ACCESS
    energy_j = compute_energy + memory_energy
    return y, flops, memory_bytes, energy_j

# --- Optimization Utilities ---

def energy_scale_for_bits(bits):
    """Scale energy per MAC/ADD based on bit-width (rough heuristic: quadratic with bits)."""
    base_bits = 32
    scale = (bits / base_bits) ** 2
    return scale

def quantize_tensor(tensor, bits=8):
    """Uniform symmetric quantization to given bit-width. Returns quantized tensor and scale."""
    # Flatten to find max
    values = []
    if isinstance(tensor[0], list):
        for row in tensor:
            for v in row:
                values.append(v)
    else:
        for v in tensor:
            values.append(v)
    max_abs = max([abs(v) for v in values]) if values else 1.0
    qmax = (1 << (bits - 1)) - 1
    scale = max_abs / qmax if qmax > 0 else 1.0
    
    def q(v):
        if scale == 0:
            return 0
        qv = int(round(v / scale))
        qv = max(-qmax-1, min(qv, qmax))
        return qv
    
    if isinstance(tensor[0], list):
        qt = [[q(v) for v in row] for row in tensor]
    else:
        qt = [q(v) for v in tensor]
    
    # Cost model: reduced energy per op
    scale_factor = energy_scale_for_bits(bits)
    energy_per_mac = ENERGY_PER_MAC_FP32 * scale_factor
    energy_per_add = ENERGY_PER_ADD_FP32 * scale_factor
    return qt, scale, energy_per_mac, energy_per_add

def dequantize_tensor(qtensor, scale):
    if isinstance(qtensor[0], list):
        return [[v * scale for v in row] for row in qtensor]
    return [v * scale for v in qtensor]

def prune_kernel(kernel, sparsity=0.5):
    """Structured pruning on kernel weights by zeroing smallest magnitudes to reach sparsity."""
    H, W = len(kernel), len(kernel[0])
    flat = []
    for i in range(H):
        for j in range(W):
            flat.append(abs(kernel[i][j]))
    k = int(len(flat) * sparsity)
    threshold = 0
    if k > 0 and len(flat) > 0:
        sorted_vals = sorted(flat)
        threshold = sorted_vals[k-1]
    pruned = []
    for i in range(H):
        row = []
        for j in range(W):
            v = kernel[i][j]
            row.append(0 if abs(v) <= threshold else v)
        pruned.append(row)
    return pruned

def kmeans_1d(values, K, iters=15):
    """Simple 1D k-means clustering for weight sharing. Returns (centroids, assignments).
    values: list[float]
    assignments: list[int] same length as values indicating centroid index.
    """
    if not values:
        return [0.0], []
    # Initialize centroids by picking K spread quantiles
    sorted_vals = sorted(values)
    if K <= 0:
        K = 1
    if K > len(sorted_vals):
        K = len(sorted_vals)
    centroids = []
    for k in range(K):
        q_idx = int((k + 0.5) * len(sorted_vals) / K)
        q_idx = min(max(q_idx, 0), len(sorted_vals) - 1)
        centroids.append(sorted_vals[q_idx])
    assignments = [0] * len(values)
    for _ in range(max(1, iters)):
        # Assign
        for i, v in enumerate(values):
            best_k = 0
            best_d = float('inf')
            for k, c in enumerate(centroids):
                d = (v - c) * (v - c)
                if d < best_d:
                    best_d = d
                    best_k = k
            assignments[i] = best_k
        # Update
        sums = [0.0] * K
        counts = [0] * K
        for v, a in zip(values, assignments):
            sums[a] += v
            counts[a] += 1
        for k in range(K):
            if counts[k] > 0:
                centroids[k] = sums[k] / counts[k]
    return centroids, assignments

def cluster_kernel(kernel, K):
    """Cluster a 2D kernel KxK into a codebook representation.
    Returns dict: {"codebook": list[float], "indices": 2D list}
    """
    H, W = len(kernel), len(kernel[0])
    vals = []
    for i in range(H):
        for j in range(W):
            vals.append(kernel[i][j])
    codebook, assigns = kmeans_1d(vals, K)
    idxs2d = []
    t = 0
    for i in range(H):
        row = []
        for j in range(W):
            row.append(assigns[t])
            t += 1
        idxs2d.append(row)
    return {"codebook": codebook, "indices": idxs2d}

def cluster_fc_weights(weights, K):
    """Cluster a 2D FC weight matrix (N_out x N_in) into codebook form.
    Returns dict: {"codebook": list[float], "indices": 2D list N_out x N_in}
    """
    N_out = len(weights)
    N_in = len(weights[0]) if N_out > 0 else 0
    vals = []
    for o in range(N_out):
        for i in range(N_in):
            vals.append(weights[o][i])
    codebook, assigns = kmeans_1d(vals, K)
    idxs2d = []
    t = 0
    for o in range(N_out):
        row = []
        for i in range(N_in):
            row.append(assigns[t])
            t += 1
        idxs2d.append(row)
    return {"codebook": codebook, "indices": idxs2d}

def reconstruct_from_codebook(obj):
    """Reconstruct 2D matrix from codebook object {codebook, indices}."""
    cb = obj['codebook']
    idxs = obj['indices']
    H = len(idxs)
    W = len(idxs[0]) if H > 0 else 0
    out = []
    for i in range(H):
        row = []
        for j in range(W):
            row.append(cb[idxs[i][j]])
        out.append(row)
    return out

def low_rank_factorize_2d(kernel, rank=1, iters=5):
    """Low-rank factorization using simple alternating minimization (rank-1 by default)."""
    K = len(kernel)
    # Initialize column vector c as first column
    c = [kernel[i][0] for i in range(K)]
    for _ in range(iters):
        # Solve for r given c: minimize ||K - c r^T||_F => r_j = (c^T k_j)/(c^T c)
        denom = sum([ci*ci for ci in c]) or 1.0
        r = []
        for j in range(K):
            num = 0
            for i in range(K):
                num += c[i] * kernel[i][j]
            r.append(num / denom)
        # Solve for c given r
        denom_r = sum([rj*rj for rj in r]) or 1.0
        new_c = []
        for i in range(K):
            num = 0
            for j in range(K):
                num += kernel[i][j] * r[j]
            new_c.append(num / denom_r)
        c = new_c
    return c, r  # kernel ≈ outer(c, r)

def conv2d_winograd3x3_analytical(input_map, kernel, stride=1, padding=0):
    """
    Computes output using standard conv for correctness but reports FLOPs based on Winograd F(2x2,3x3) analytical savings.
    Note: stride must be 1 for Winograd tiling.
    """
    out, flops_naive, mem_bytes, energy_naive = conv2d(input_map, kernel, stride=stride, padding=padding)
    if stride != 1:
        return out, flops_naive, mem_bytes, energy_naive
    H_out, W_out = len(out), len(out[0])
    # Winograd reduces multiplications. For F(2x2,3x3), multiplications per 2x2 tile ~ 16 vs 36 naive.
    tiles_h = H_out // 2
    tiles_w = W_out // 2
    macs_naive = H_out * W_out * 9
    macs_winograd = tiles_h * tiles_w * 16
    # Handle edge tiles (fallback to naive estimate)
    leftover = (H_out * W_out) - (tiles_h * 2) * (tiles_w * 2)
    macs_winograd += leftover * 9
    # FLOPs count: 2 per MAC (mul+add) approximately
    flops = macs_winograd * 2
    # Energy scaling assumes same memory, reduced compute
    compute_energy = macs_winograd * ENERGY_PER_MAC_FP32
    energy_j = compute_energy + (mem_bytes * ENERGY_PER_BYTE_ACCESS)
    return out, flops, mem_bytes, energy_j

def conv2d_fft_analytical(input_map, kernel, stride=1, padding=0):
    """
    Computes output via standard conv but reports FFT-based convolution cost analytically.
    Useful for larger K.
    """
    out, flops_naive, mem_bytes, energy_naive = conv2d(input_map, kernel, stride=stride, padding=padding)
    H_in = len(input_map)
    W_in = len(input_map[0])
    K = len(kernel)
    H_out = len(out)
    W_out = len(out[0])
    # Analytical complexity: using 2D FFTs: ~ C * (H_in W_in log(H_in W_in)) ignoring constants, plus elementwise mult and iFFT.
    # We'll estimate flops as a reduced factor vs naive when K is large.
    N = max(H_in, W_in)
    # crude log2
    def log2_int(n):
        p = 0
        while (1 << p) < n:
            p += 1
        return p
    logN = log2_int(N)
    flops_fft = 10 * H_in * W_in * logN  # constant 10 is heuristic for operations per element per stage
    mults = H_in * W_in  # elementwise mult in frequency
    flops = int(flops_fft + 2 * mults)
    compute_energy = (flops // 2) * ENERGY_PER_MAC_FP32
    energy_j = compute_energy + (mem_bytes * ENERGY_PER_BYTE_ACCESS)
    return out, flops, mem_bytes, energy_j

def conv2d_fp16_analytical(input_map, kernel, stride=1, padding=0):
    """
    Executes standard conv for correctness but scales compute energy as FP16.
    Returns (output_map, flops, memory_bytes, energy_j)
    """
    out, flops, mem_bytes, _ = conv2d(input_map, kernel, stride=stride, padding=padding)
    # Scale energy per MAC by FP16 bit-width model
    scale_factor = energy_scale_for_bits(16)
    num_macs = (flops // 2)  # approx MACs
    compute_energy = num_macs * (ENERGY_PER_MAC_FP32 * scale_factor)
    energy_j = compute_energy + (mem_bytes * ENERGY_PER_BYTE_ACCESS)
    return out, flops, mem_bytes, energy_j

# ---------------- Polynomial activation approximation ----------------

def poly_activation(input_map, coeffs=None, kind='tanh3'):
    """
    Evaluate a polynomial activation y = c0 + c1 x + c2 x^2 + ... on each element.
    Predefined kinds:
      - 'sigmoid3': ~ 0.5 + 0.2159198 x - 0.0082176 x^3 (good on [-3,3])
      - 'tanh3':    ~ 0.0 + 0.9926 x - 0.26037 x^3 (rough Chebyshev fit)
      - 'relu3':    cubic smooth ReLU approx: 0.5 x + 0.125 x^3, clipped >=0
    Returns (output_map, flops, memory_bytes, energy_j)
    """
    if coeffs is None:
        if kind == 'sigmoid3':
            coeffs = [0.5, 0.2159198, 0.0, -0.0082176]
        elif kind == 'tanh3':
            coeffs = [0.0, 0.9926, 0.0, -0.26037]
        elif kind == 'relu3':
            coeffs = [0.0, 0.5, 0.0, 0.125]
        else:
            coeffs = [0.0, 1.0]  # identity
    H, W = len(input_map), len(input_map[0])
    deg = len(coeffs) - 1
    out = [[0.0]*W for _ in range(H)]
    flops = 0
    # Horner's method per element
    for i in range(H):
        for j in range(W):
            x = input_map[i][j]
            y = 0.0
            for k in range(deg, -1, -1):
                y = y * x + coeffs[k]
                # mul + add except first assignment -> count approx 2 each iter
                flops += 2
            if kind == 'relu3':
                # ensure non-negativity as ReLU-like
                y = y if y > 0 else 0.0
                flops += 1
            out[i][j] = y
    memory_bytes = (H*W*4) + (H*W*4)
    energy_j = (flops * ENERGY_PER_ADD_FP32) + (memory_bytes * ENERGY_PER_BYTE_ACCESS)
    return out, flops, memory_bytes, energy_j

# ---------------- Strassen matrix multiplication for dense layers ----------------

def _add_mat(A, B):
    n = len(A); m = len(A[0])
    C = [[0.0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] + B[i][j]
    return C

def _sub_mat(A, B):
    n = len(A); m = len(A[0])
    C = [[0.0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] - B[i][j]
    return C

def _naive_mm(A, B):
    n = len(A); k = len(A[0]); m = len(B[0])
    C = [[0.0]*m for _ in range(n)]
    for i in range(n):
        for p in range(k):
            a = A[i][p]
            for j in range(m):
                C[i][j] += a * B[p][j]
    return C

def _pad_to_pow2(A):
    n = len(A)
    m = len(A[0])
    def next_pow2(x):
        p = 1
        while p < x:
            p <<= 1
        return p
    N = next_pow2(n)
    M = next_pow2(m)
    out = [[0.0]*M for _ in range(N)]
    for i in range(n):
        for j in range(m):
            out[i][j] = A[i][j]
    return out, n, m, N, M

def matmul_strassen(A, B, cutoff=64):
    """
    Square-matrix Strassen with padding to powers of 2; falls back to naive at small sizes.
    Returns C = A x B and an approximate (flops, memory, energy) model.
    """
    # Pad A and B to square matrices of size SxS
    A_pad, a_n, a_k, A_N, A_M = _pad_to_pow2(A)
    B_pad, b_n, b_m, B_N, B_M = _pad_to_pow2(B)
    S = max(A_N, A_M, B_N, B_M)
    # Extend to SxS
    def extend(Mtx, S):
        n = len(Mtx); m = len(Mtx[0])
        out = [[0.0]*S for _ in range(S)]
        for i in range(n):
            for j in range(m):
                out[i][j] = Mtx[i][j]
        return out
    A_ext = extend(A_pad, S)
    B_ext = extend(B_pad, S)

    flops = 0

    def strassen(X, Y):
        nonlocal flops
        n = len(X)
        if n <= cutoff:
            C = _naive_mm(X, Y)
            # naive flops ~ 2*n^3
            flops += 2 * n * n * n
            return C
        m = n // 2
        # Split matrices into quadrants
        A11 = [row[:m] for row in X[:m]]
        A12 = [row[m:] for row in X[:m]]
        A21 = [row[:m] for row in X[m:]]
        A22 = [row[m:] for row in X[m:]]
        B11 = [row[:m] for row in Y[:m]]
        B12 = [row[m:] for row in Y[:m]]
        B21 = [row[:m] for row in Y[m:]]
        B22 = [row[m:] for row in Y[m:]]
        # 7 products
        M1 = strassen(_add_mat(A11, A22), _add_mat(B11, B22))
        M2 = strassen(_add_mat(A21, A22), B11)
        M3 = strassen(A11, _sub_mat(B12, B22))
        M4 = strassen(A22, _sub_mat(B21, B11))
        M5 = strassen(_add_mat(A11, A12), B22)
        M6 = strassen(_sub_mat(A21, A11), _add_mat(B11, B12))
        M7 = strassen(_sub_mat(A12, A22), _add_mat(B21, B22))
        # Combine
        C11 = _add_mat(_sub_mat(_add_mat(M1, M4), M5), M7)
        C12 = _add_mat(M3, M5)
        C21 = _add_mat(M2, M4)
        C22 = _add_mat(_sub_mat(_add_mat(M1, M3), M2), M6)
        # Assemble C
        C = [[0.0]*n for _ in range(n)]
        for i in range(m):
            C[i][:m] = C11[i]
            C[i][m:] = C12[i]
        for i in range(m):
            C[m+i][:m] = C21[i]
            C[m+i][m:] = C22[i]
        # Add flop estimates for additions: each matrix add/sub ~ n^2
        add_ops = 18 * (m * m)  # heuristic for number of adds/subs on m x m blocks
        flops += add_ops
        return C

    C_ext = strassen(A_ext, B_ext)
    # Crop to original shape (a_n x b_m)
    C = [row[:b_m] for row in C_ext[:a_n]]
    # Memory and energy estimates
    memory_bytes = (
        (a_n * a_k * 4) + (b_n * b_m * 4) + (a_n * b_m * 4)
    )
    compute_energy = (flops // 2) * ENERGY_PER_MAC_FP32
    energy_j = compute_energy + (memory_bytes * ENERGY_PER_BYTE_ACCESS)
    return C, flops, memory_bytes, energy_j

def fully_connected_strassen(x_vec, weights, biases=None):
    """
    FC using Strassen matrix multiplication. weights: (N_out x N_in), x_vec: (N_in)
    Returns (y_vec, flops, memory_bytes, energy_j)
    """
    N_out = len(weights)
    N_in = len(weights[0]) if N_out > 0 else 0
    # Convert vector to N_in x 1 matrix
    X = [[x] for x in x_vec]
    Y, flops, mem_bytes, energy = matmul_strassen(weights, X)
    y = [row[0] for row in Y]
    # Add bias cost
    if biases is not None:
        for o in range(N_out):
            y[o] += biases[o]
        flops += N_out
        mem_bytes += N_out * 4
        energy += (N_out * ENERGY_PER_ADD_FP32) + (N_out * 4 * ENERGY_PER_BYTE_ACCESS)
    return y, flops, mem_bytes, energy

# ---------------- Binarization / Ternarization ----------------

def binarize_weights(matrix):
    """Binarize weights to {-1, +1} with scale alpha = mean(|w|)."""
    vals = []
    for row in matrix:
        for v in row:
            vals.append(abs(v))
    alpha = (sum(vals) / len(vals)) if vals else 1.0
    binW = [[1 if w >= 0 else -1 for w in row] for row in matrix]
    return binW, alpha

def ternarize_weights(matrix, threshold_ratio=0.05):
    """Ternarize weights to {-1, 0, +1}; zero small magnitudes. Alpha is mean |w| of nonzeros."""
    # Compute threshold as ratio of max |w|
    mags = [abs(w) for row in matrix for w in row]
    t = (max(mags) * threshold_ratio) if mags else 0.0
    binW = []
    nz_vals = []
    for row in matrix:
        bro = []
        for w in row:
            if abs(w) < t:
                bro.append(0)
            else:
                bro.append(1 if w >= 0 else -1)
                nz_vals.append(abs(w))
        binW.append(bro)
    alpha = (sum(nz_vals) / len(nz_vals)) if nz_vals else 1.0
    return binW, alpha

def fully_connected_binarized(x_vec, weights, biases=None, ternary=False):
    """FC with binarized/ternarized weights and scale alpha. Returns resource estimates."""
    if ternary:
        binW, alpha = ternarize_weights(weights)
    else:
        binW, alpha = binarize_weights(weights)
    N_out = len(binW)
    N_in = len(binW[0]) if N_out > 0 else 0
    y = [0.0]*N_out
    flops = 0
    # In binary case, mul reduces to add/sub; we still count as 1 add per nonzero sign
    for o in range(N_out):
        acc = 0.0
        for i in range(N_in):
            s = binW[o][i]
            if s == 0:
                continue
            acc += (x_vec[i] if s > 0 else -x_vec[i])
            flops += 1
        acc *= alpha
        flops += 1
        if biases is not None:
            acc += biases[o]
            flops += 1
        y[o] = acc
    memory_bytes = (N_in*4) + (N_out*N_in*1) + (N_out*4)  # approx: 1 byte per bin weight
    energy_j = (flops * ENERGY_PER_ADD_FP32) + (memory_bytes * ENERGY_PER_BYTE_ACCESS)
    return y, flops, memory_bytes, energy_j

def conv2d_binarized(input_map, kernel, stride=1, padding=0, bias=0.0, ternary=False):
    """Binary/Ternary weight conv: sign weights with scale alpha; reduces muls to add/sub."""
    binK, alpha = ternarize_weights(kernel) if ternary else binarize_weights(kernel)
    # Standard conv loop but using signs
    H_in = len(input_map); W_in = len(input_map[0])
    K = len(binK)
    if padding > 0:
        padded = [[0]*(W_in+2*padding) for _ in range(H_in+2*padding)]
        for i in range(H_in):
            for j in range(W_in):
                padded[i+padding][j+padding] = input_map[i][j]
        input_map = padded
        H_in += 2*padding; W_in += 2*padding
    H_out = (H_in - K)//stride + 1
    W_out = (W_in - K)//stride + 1
    out = [[0.0]*W_out for _ in range(H_out)]
    flops = 0
    for i in range(H_out):
        for j in range(W_out):
            acc = 0.0
            for ki in range(K):
                for kj in range(K):
                    s = binK[ki][kj]
                    if s == 0:
                        continue
                    v = input_map[i*stride+ki][j*stride+kj]
                    acc += (v if s > 0 else -v)
                    flops += 1
            acc = acc * alpha + bias
            flops += 2
            out[i][j] = acc
    memory_bytes = (H_in*W_in*4) + (K*K*1) + (H_out*W_out*4)
    energy_j = (flops * ENERGY_PER_ADD_FP32) + (memory_bytes * ENERGY_PER_BYTE_ACCESS)
    return out, flops, memory_bytes, energy_j

# ---------------- Channel pruning utilities ----------------

def channel_l1_norm(kernel_ch):
    """L1 norm of a single channel kernel (KxK)."""
    s = 0.0
    K = len(kernel_ch)
    for i in range(K):
        for j in range(K):
            s += abs(kernel_ch[i][j])
    return s

def prune_channels(kernels, keep_ratio=0.5):
    """
    Given a list of per-channel kernels (list[C][K][K]), keep top channels by L1 norm.
    Returns (pruned_kernels, kept_indices)
    """
    C = len(kernels)
    norms = [(channel_l1_norm(kernels[c]), c) for c in range(C)]
    norms.sort(reverse=True)
    k = max(1, int(C * keep_ratio))
    kept = sorted([idx for _, idx in norms[:k]])
    pruned = [kernels[c] for c in kept]
    return pruned, kept

def apply_channel_pruning(input_tensor, kernels, keep_ratio=0.5):
    """
    Prune input channels and corresponding kernels consistently.
    input_tensor: list[C][H][W], kernels: list[C][K][K]
    Returns (pruned_input_tensor, pruned_kernels, kept_indices)
    """
    pruned_k, kept = prune_channels(kernels, keep_ratio=keep_ratio)
    pruned_input = [input_tensor[c] for c in kept]
    return pruned_input, pruned_k, kept

def entropy(values):
    # values: list of numbers; compute discrete entropy of magnitude histogram (10 bins)
    if not values:
        return 0.0
    mags = [abs(v) for v in values]
    max_v = max(mags) or 1.0
    bins = [0]*10
    for v in mags:
        idx = int((v / max_v) * 9)
        bins[idx] += 1
    total = float(len(mags))
    H = 0.0
    for b in bins:
        if b > 0:
            p = b/total
            H -= p * math.log(p + 1e-12, 2)
    return H

def entropy_prune_kernel(kernel, threshold_bits=1.0):
    """Remove (zero) kernel if entropy of weights below threshold (in bits)."""
    vals = []
    for row in kernel:
        for v in row:
            vals.append(v)
    H = entropy(vals)
    if H < threshold_bits:
        return [[0 for _ in row] for row in kernel]
    return kernel

def main():
    """Main function to run a toy example."""
    print("--- Pure Python CNN Toy Example ---")

    # Define a dummy input (5x5, 1 channel)
    dummy_input = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ]

    # Define a 3x3 kernel
    kernel = [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ]

    print("Input Map (5x5):")
    for row in dummy_input:
        print(row)
    
    print("\nKernel (3x3):")
    for row in kernel:
        print(row)

    # --- 1. Convolutional Layer ---
    print("\n--- 1. Applying Convolution (3x3 kernel, stride=1) ---")
    conv_output, conv_flops, conv_mem, conv_energy = conv2d(dummy_input, kernel, stride=1, padding=0)
    
    print("Convolution Output:")
    for row in conv_output:
        print([f"{x:4.0f}" for x in row])
    
    print(f"  - FLOPs: {conv_flops}")
    print(f"  - Memory: {conv_mem} bytes")
    print(f"  - Energy: {conv_energy:.2e} J")

    # --- 2. ReLU ---
    print("\n--- 2. ReLU Activation ---")
    relu_out, relu_flops, relu_mem, relu_energy = relu(conv_output)
    for row in relu_out:
        print([f"{x:4.0f}" for x in row])
    print(f"  - FLOPs: {relu_flops}")
    print(f"  - Memory: {relu_mem} bytes")
    print(f"  - Energy: {relu_energy:.2e} J")

    # --- 3. Max Pooling (2x2, stride 2) ---
    print("\n--- 3. Max Pooling (2x2, stride=2) ---")
    pool_out, pool_flops, pool_mem, pool_energy = max_pool2d(relu_out, pool_size=2, stride=2)
    for row in pool_out:
        print([f"{x:4.0f}" for x in row])
    print(f"  - FLOPs: {pool_flops}")
    print(f"  - Memory: {pool_mem} bytes")
    print(f"  - Energy: {pool_energy:.2e} J")

    # Average Pooling demonstration (2x2, stride 2)
    print("\n--- 3b. Average Pooling (2x2, stride=2) ---")
    avg_out, avg_flops, avg_mem, avg_energy = avg_pool2d(relu_out, pool_size=2, stride=2)
    for row in avg_out:
        print([f"{x:4.0f}" for x in row])
    print(f"  - FLOPs: {avg_flops}")
    print(f"  - Memory: {avg_mem} bytes")
    print(f"  - Energy: {avg_energy:.2e} J")

    # --- 4. Flatten ---
    print("\n--- 4. Flatten ---")
    flat, flat_flops, flat_mem, flat_energy = flatten(pool_out)
    print(f"Flat length: {len(flat)} -> {flat}")
    print(f"  - FLOPs: {flat_flops}")
    print(f"  - Memory: {flat_mem} bytes")
    print(f"  - Energy: {flat_energy:.2e} J")

    # --- 5. Fully Connected to 2 outputs ---
    print("\n--- 5. Fully Connected (to 2 outputs) ---")
    N_in = len(flat)
    weights = [[0.1 for _ in range(N_in)], [0.2 for _ in range(N_in)]]
    biases = [0.5, -0.3]
    fc_out, fc_flops, fc_mem, fc_energy = fully_connected(flat, weights, biases)
    print(f"FC Output: {[round(v,3) for v in fc_out]}")
    print(f"  - FLOPs: {fc_flops}")
    print(f"  - Memory: {fc_mem} bytes")
    print(f"  - Energy: {fc_energy:.2e} J")

    # --- 5b. Weight Clustering (Codebook) Demo ---
    print("\n--- 5b. Weight Clustering (Codebook) Demo ---")
    Kc = 4  # small codebook size
    clustered_fc = cluster_fc_weights(weights, Kc)
    fc_out_cb, fc_flops_cb, fc_mem_cb, fc_energy_cb = fully_connected(flat, clustered_fc, biases)
    print(f"Codebook size: {len(clustered_fc['codebook'])}")
    print(f"FC Output (clustered): {[round(v,3) for v in fc_out_cb]}")
    print(f"  - FLOPs: {fc_flops_cb}")
    print(f"  - Memory (clustered): {fc_mem_cb} bytes (vs {fc_mem} bytes)")
    print(f"  - Energy (clustered): {fc_energy_cb:.2e} J (vs {fc_energy:.2e} J)")

    # --- 1b. Clustered conv kernel demo ---
    print("\n--- 1b. Clustered Conv Kernel Demo ---")
    clustered_k = cluster_kernel(kernel, K=4)
    conv_output_cb, conv_flops_cb, conv_mem_cb, conv_energy_cb = conv2d(dummy_input, clustered_k, stride=1, padding=0)
    print("Clustered kernel codebook:", [round(c,3) for c in clustered_k['codebook']])
    print(f"  - Conv Memory (clustered): {conv_mem_cb} bytes (vs {conv_mem} bytes)")
    print(f"  - Conv Energy (clustered): {conv_energy_cb:.2e} J (vs {conv_energy:.2e} J)")

    # --- Aggregated metrics ---
    total_flops = conv_flops + relu_flops + pool_flops + flat_flops + fc_flops
    total_mem = conv_mem + relu_mem + pool_mem + flat_mem + fc_mem
    total_energy = conv_energy + relu_energy + pool_energy + flat_energy + fc_energy
    print("\n=== Aggregated Metrics ===")
    print(f"Total FLOPs: {total_flops}")
    print(f"Total Memory (bytes): {total_mem}")
    print(f"Total Energy (J): {total_energy:.2e}")

    # --- Demonstrate Optimizations (analytical) ---
    print("\n=== Optimizations (Analytical Demonstrations) ===")
    # Quantization of kernel to INT8
    qk, qscale, e_mac_q, e_add_q = quantize_tensor(kernel, bits=8)
    print(f"Quantized kernel (INT8), scale={qscale:.4f}: {qk}")
    # Recompute conv output with dequantized kernel (same numerically), but energy scaled
    dq_kernel = dequantize_tensor(qk, qscale)
    conv_out_q, conv_flops_q, conv_mem_q, _ = conv2d(dummy_input, dq_kernel, stride=1, padding=0)
    # Compute energy with reduced per-MAC energy
    H_out_q, W_out_q = len(conv_out_q), len(conv_out_q[0])
    macs_q = H_out_q * W_out_q * 9
    compute_energy_q = macs_q * e_mac_q
    energy_q = compute_energy_q + (conv_mem_q * ENERGY_PER_BYTE_ACCESS)
    print(f"Quantized Conv Energy Estimate: {energy_q:.2e} J (vs {conv_energy:.2e} J)")

    # Pruning
    pruned_k = prune_kernel(kernel, sparsity=0.5)
    conv_out_p, conv_flops_p, conv_mem_p, conv_energy_p = conv2d(dummy_input, pruned_k, stride=1, padding=0)
    print(f"Pruned kernel (50%): {pruned_k}")
    print(f"Pruned Conv Energy: {conv_energy_p:.2e} J")

    # Low-rank factorization (rank-1)
    c, r = low_rank_factorize_2d(kernel, rank=1, iters=5)
    # Implement separable conv via two 1D convs analytically: cost ~ H_out*W_out*(2*K) MACs instead of 9
    conv_lr_out, _, _, _ = conv2d(dummy_input, kernel, stride=1, padding=0)
    H_out_lr, W_out_lr = len(conv_lr_out), len(conv_lr_out[0])
    macs_lr = H_out_lr * W_out_lr * (2 * len(kernel))
    energy_lr = macs_lr * ENERGY_PER_MAC_FP32 + (conv_mem * ENERGY_PER_BYTE_ACCESS)
    print(f"Low-rank approx vectors len={len(c)}: c={[(round(x,3)) for x in c]}, r={[(round(x,3)) for x in r]}")
    print(f"Low-rank Energy Estimate: {energy_lr:.2e} J")

    # Winograd analytical for 3x3
    _, flops_win, mem_win, energy_win = conv2d_winograd3x3_analytical(dummy_input, kernel)
    print(f"Winograd (analytical) Energy: {energy_win:.2e} J")

    # FFT analytical for larger kernels (simulate with current K=3 as example)
    _, flops_fft, mem_fft, energy_fft = conv2d_fft_analytical(dummy_input, kernel)
    print(f"FFT-based (analytical) Energy: {energy_fft:.2e} J")

    # Entropy-based pruning demonstration
    ent_k = entropy_prune_kernel(kernel, threshold_bits=0.5)
    _, _, _, ent_energy = conv2d(dummy_input, ent_k, stride=1, padding=0)
    ent_val = entropy([v for row in kernel for v in row])
    print(f"Entropy of original kernel: {ent_val:.3f} bits")
    print(f"Entropy-pruned kernel (thr=0.5): {ent_k}")
    print(f"Entropy-pruned Conv Energy: {ent_energy:.2e} J")

    # --- Multi-channel convolution demo ---
    print("\n=== Multi-channel Convolution Demo ===")
    mc_input = [
        dummy_input,  # ch0
        [[v*0.5 for v in row] for row in dummy_input]  # ch1 scaled
    ]
    mc_kernels = [
        kernel,
        [[0, 1, 0],[0, 1, 0],[0, 1, 0]]
    ]
    mc_out, mc_flops, mc_mem, mc_energy = conv2d_multi(mc_input, mc_kernels, stride=1, padding=0, bias=0.1)
    for row in mc_out:
        print([f"{x:5.2f}" for x in row])
    print(f"  - FLOPs: {mc_flops}")
    print(f"  - Memory: {mc_mem} bytes")
    print(f"  - Energy: {mc_energy:.2e} J")

    # --- Benchmark test harness: naive vs Winograd vs FFT (analytical) ---
    print("\n=== Benchmark: Naive vs Winograd vs FFT (analytical) ===")
    sizes = [3, 5, 7]
    bench_input = [[(i+j)%5 for j in range(16)] for i in range(16)]
    for K in sizes:
        # simple kernel with central 1s pattern
        ker = [[0 for _ in range(K)] for _ in range(K)]
        for t in range(K):
            ker[t][t] = 1
        out_n, flops_n, mem_n, en_n = conv2d(bench_input, ker, stride=1, padding=0)
        _, flops_w, mem_w, en_w = conv2d_winograd3x3_analytical(bench_input, ker) if K == 3 else (out_n, flops_n, mem_n, en_n)
        _, flops_f, mem_f, en_f = conv2d_fft_analytical(bench_input, ker)
        print(f"K={K}: Naive FLOPs={flops_n}, Energy={en_n:.2e} | Winograd FLOPs={(flops_w if K==3 else 'n/a')} Energy={(f'{en_w:.2e}' if K==3 else 'n/a')} | FFT FLOPs={flops_f}, Energy={en_f:.2e}")

if __name__ == "__main__":
    run_cli(sys.argv)
