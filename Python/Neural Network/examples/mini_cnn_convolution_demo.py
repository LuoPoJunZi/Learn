"""A tiny convolution demo inspired by CNNs.

This is not a full CNN training example. It focuses on one key idea:

    convolution kernels can extract local patterns from a grid.

The input is a small 5x5 image-like matrix. Two kernels are applied:

- vertical edge kernel
- horizontal edge kernel
"""


def convolve2d(image, kernel):
    image_height = len(image)
    image_width = len(image[0])
    kernel_size = len(kernel)
    output_size = image_height - kernel_size + 1

    output = []
    for row in range(output_size):
        output_row = []
        for col in range(output_size):
            total = 0
            for kernel_row in range(kernel_size):
                for kernel_col in range(kernel_size):
                    total += image[row + kernel_row][col + kernel_col] * kernel[kernel_row][kernel_col]
            output_row.append(total)
        output.append(output_row)
    return output


def print_matrix(title, matrix):
    print(title)
    for row in matrix:
        print(" ".join(f"{value:>4}" for value in row))
    print()


def main():
    image = [
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ]

    vertical_edge_kernel = [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ]

    horizontal_edge_kernel = [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
    ]

    print_matrix("Input image:", image)
    print_matrix("Vertical edge response:", convolve2d(image, vertical_edge_kernel))
    print_matrix("Horizontal edge response:", convolve2d(image, horizontal_edge_kernel))


if __name__ == "__main__":
    main()

