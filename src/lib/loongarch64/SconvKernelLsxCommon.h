/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    SconvKernelLsxCommon.h

Abstract:

    This module contains common kernel macros and structures for the single
    precision convolution operation for the Lsx kernels.

--*/

#define SP_SIZE 32*8

#define MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT     0x00000001
#define MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION         0x00000002
#define MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION       0x00000004
#define MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION      0x00000008

#define Filter_save_offset 18*8

#define OutputStride_arg                6*8
#define KernelHeight_arg                7*8
#define KernelWidth_arg                 8*8
#define InputBase_arg                   9*8
#define InputWidth_arg                  10*8
#define DilatedInputWidth_arg           11*8
#define OutputCountLeftPad_arg          12*8
#define OutputCount_arg                 13*8
#define OutputCountRightPad_arg         14*8
#define Bias_arg                        15*8
#define Flags_arg                       16*8
#define InputChannels_arg               17*8

/*++

Macro Description:

    This macro generates code to compute the convolution for a vector of input
    blocks and a vector of filter blocks to produce a matrix of output blocks.

    OutputCount=1 generates special case code to handle padding blocks. All
    other output counts assume no padding.

Arguments:

    Isa - Supplies the instruction set architecture string for function tags.

    KernelFrame - Supplies the symbol name to access the convolution kernel
        stack.

    KernelType - Supplies the type of kernel to be generated.

    BlockSize - Supplies the number of elements per block.

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    a0 - Supplies the address of the input buffer.

    a1 - Supplies the FilterStride parameter (see function description) when
        KernelType!=Depthwise. Supplies the address of the filter buffer when
        KernelType=Depthwise.

    s8 - Supplies the DilationWidth parameter (see function description).

    a4 - Supplies the address of the output buffer.

    a5 - Supplies the StrideWidth parameter (see function description).

    s3 - Supplies the InputStride parameter (see function description).
--*/

        .macro ProcessOutputCountN Isa, KernelFrame, KernelType, BlockSize, FilterCount, OutputCount
        move    $a3, $a0
.ifeqs "\KernelType\()","Depthwise"
        move     $a2, $a1
.else
        ld.d    $a2, $sp, Filter_save_offset
.endif
        ld.d    $t1, $sp, KernelHeight_arg   //KernelHeight
        ld.d    $t2, $sp, KernelWidth_arg   //KernelWidth
.if \OutputCount\() == 1
        ld.d    $t3, $sp, InputBase_arg   //InputBase
        ld.d    $t4, $sp, InputWidth_arg   //InputWidth
        sub.d   $t3, $zero, $t3                         # keep negative for lea usage below
.endif
        ClearBlock \FilterCount\(), \OutputCount\()
        beqz    $t1, .L\KernelType\().\FilterCount\().\OutputCount\().HandlePostProcessing

.L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextRow:
        move     $t6, $t2                     # reload kernel width remaining
.L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextColumn:
.if \OutputCount\() == 1
        add.d   $t7, $a3, $t3
        bgeu     $t7, $t4, .L\KernelType\().\FilterCount\().\OutputCount\().SkipOverPadding
.endif
.if \OutputCount\() > 3
        li.d    $s2, 2
        mul.d   $s2, $a5, $s2
        add.d   $t4, $a5, $s2

        add.d   $t4, $t4, $a3                # compute input plus 3 blocks
.endif
.if \FilterCount\() > 2
        li.d    $s2, 2
        mul.d   $s2, $s2, $a1
        add.d   $t7, $a2, $s2       //t6 is rbx used by ComputeBlock
.endif
.ifeqs "\KernelType\()","Nchwc"
.if \BlockSize\() == 16
        .irp Index, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            ComputeBlock \KernelType\(), \FilterCount\(), \OutputCount\(), \Index\()*16*4, \Index\()*4
        .endr
.else
        .irp Index, 0, 1, 2, 3, 4, 5, 6, 7
            ComputeBlock \KernelType\(), \FilterCount\(), \OutputCount\(), (\Index\()-4)*8*4, \Index\()*4
        .endr
.endif
.else
        ComputeBlock \KernelType\(), \FilterCount\(), \OutputCount\(), 0, 0
.endif
.L\KernelType\().\FilterCount\().\OutputCount\().SkipOverPadding:
        add.d   $a3, $a3, $t8               # advance input by dilation width
.ifeqs "\KernelType\()","Nchwc"
        addi.d  $a2, $a2, \BlockSize\()*\BlockSize\()*4
                                            # advance filter by 8i8o/16i16o block
.else
        addi.d  $a2, $a2, \BlockSize\()*4   # advance filter by 8o/16o block
.endif
        addi.d  $t6, $t6, -1                # decrement columns remaining
        bnez    $t6,    .L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextColumn
        add.d   $a3, $a3, $t5
.if \OutputCount\() == 1
        ld.d    $s0, $sp, DilatedInputWidth_arg            #DilatedInputWidth
        sub.d   $t3, $t3, $s0
                                            # advance input base to next row
.endif
        addi.d  $t1, $t1, -1                         # decrement rows remaining
        bnez    $t1, .L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextRow

//
// Handle post processing of the output block.
//
.L\KernelType\().\FilterCount\().\OutputCount\().HandlePostProcessing:
        ld.w    $a2, $sp, Flags_arg

.if \FilterCount\() > 1
        ld.d    $t6, $sp, OutputStride_arg
.endif
        ld.d    $a3, $sp, Bias_arg
        bl    MlasConvPostProcessFloat\Isa\()Filter\FilterCount\()Output\OutputCount\()
.endm
/*++

Macro Description:

    This macro generates code for the inner convolution kernel.

Arguments:

    KernelType - Supplies the type of kernel to be generated.

    BlockSize - Supplies the number of elements per block.

    Isa - Supplies the instruction set architecture string for function tags.

    BiasFilter - Supplies a non-blank value if the address of the filter buffer
        should be biased to point to the middle of a OIhw8i8o block in order to
        reduce the code size from relative byte offsets.

--*/

        .macro SconvKernelFunction KernelType, BlockSize, Isa, BiasFilter

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

Arguments:

    Input (a0) - Supplies the address of the input buffer.

        The address is biased to include padding blocks for the left width
        dimension. The address is not biased to include padding rows for the
        left height dimension  these are accounted for in the outer kernel.

    Filter (a1) - Supplies the address of the filter buffer.

    Output (a2) - Supplies the address of the output buffer.

    StrideWidth (a3) - Supplies the length in bytes of the blocked stride width.

    DilationWidth (a4) - Supplies the length in bytes of the blocked dilation
        width.

    FilterCount (a5) - Supplies the number of filters to process in this
        iteration.

    InputStride (a6) - Supplies the length in bytes to advance the input buffer to
        the next input row.

    FilterStride (a7)- Supplies the length in bytes to advance the filter buffer
        to the next set of filters.

    OutputStride (sp,8*0) - Supplies the length in bytes to advance the output buffer
        to the next output address associated with the next set of filters.

    KernelHeight (sp,8*1)- Supplies the height of the kernel to apply. This height may
        be less than the original kernel height after removing any padding
        rows.

    KernelWidth (sp, 8*2)- Supplies the width of the kernel to apply.

    InputBase (sp, 8*3)- Supplies the address of the valid input buffer.

        This parameter is similar to the Input parameter, but does not include
        the padding blocks for the left width dimension. This parameter is used
        with the following InputWidth parameter in order to validate that the
        current input buffer address in bounds and not in the left or right
        width padding region.

    InputWidth (sp, 8*4)- Supplies the length in bytes of the blocked input width.

    DilatedInputWidth (sp, 8*5)- Supplies the length in bytes to advance the input base
        buffer to the next input row including dilation.

    OutputCountLeftPad (sp, 8*6)- Supplies the number of output elements that include
        one or more padding elements from the left edge.

    OutputCount (sp, 8*7)- Supplies the number of output elements that do not include
        any padding elements.

    OutputCountRightPad (sp, 8*8)- Supplies the number of output elements that include
        one or more padding elements from the right edge.

    Bias (sp, 8*9)- Supplies the address of the bias buffer.

    Flags (sp, 8*10)- Supplies additional flags controlling the convolution operation,
        especially post calculation options.

Return Value:

    None.

--*/

    FUNCTION_ENTRY MlasConv\KernelType\()FloatKernel\Isa\()
        addi.d  $sp, $sp, -SP_SIZE
        st.d    $s0, $sp, 0*8
        st.d    $s1, $sp, 1*8
        st.d    $s2, $sp, 2*8
        st.d    $s3, $sp, 3*8
        st.d    $s4, $sp, 4*8
        st.d    $ra, $sp, 5*8
        ld.d    $s0, $sp, SP_SIZE+0*8
        ld.d    $s1, $sp, SP_SIZE+1*8
        ld.d    $s2, $sp, SP_SIZE+2*8
        ld.d    $s3, $sp, SP_SIZE+3*8
        st.d    $s0, $sp, OutputStride_arg
        st.d    $s1, $sp, KernelHeight_arg
        st.d    $s2, $sp, KernelWidth_arg
        st.d    $s3, $sp, InputBase_arg
        ld.d    $s0, $sp, SP_SIZE+4*8
        ld.d    $s1, $sp, SP_SIZE+5*8
        ld.d    $s2, $sp, SP_SIZE+6*8
        ld.d    $s3, $sp, SP_SIZE+7*8
        st.d    $s0, $sp, InputWidth_arg
        st.d    $s1, $sp, DilatedInputWidth_arg
        st.d    $s2, $sp, OutputCountLeftPad_arg
        st.d    $s3, $sp, OutputCount_arg
        ld.d    $s0, $sp, SP_SIZE+8*8
        ld.d    $s1, $sp, SP_SIZE+9*8
        ld.d    $s2, $sp, SP_SIZE+10*8
        st.d    $s0, $sp, OutputCountRightPad_arg
        st.d    $s1, $sp, Bias_arg
        st.d    $s2, $sp, Flags_arg

.ifeqs "\BiasFilter\()","BiasFilter"
        addi.d $a1, $a1,4*8*4
.endif
        st.d    $a1, $sp, Filter_save_offset       //store  Filter
        move    $a1, $a7
        move    $t5, $a6
        move    $t8, $a4    # shuffle to Win64 register usage
        move    $t1, $a5
        move    $a4, $a2
        move    $a5, $a3

        li.d    $s0, 3
        beq     $t1, $s0, .L\KernelType\().ProcessFilterCount3
        blt     $t1, $s0, .L\KernelType\().ProcessFilterCountLessThan3
        ProcessFilterCountN SconvKernelFrame, \KernelType\(), 4
        b     .L\KernelType\().ExitKernel

.L\KernelType\().ProcessFilterCount3:
        ProcessFilterCountN SconvKernelFrame, \KernelType\(), 3
        b     .L\KernelType\().ExitKernel

.L\KernelType\().ProcessFilterCountLessThan3:
        li.d     $s0,2
        blt      $t1, $s0, .L\KernelType\().ProcessFilterCount1
        ProcessFilterCountN SconvKernelFrame, \KernelType\(), 2
        b     .L\KernelType\().ExitKernel

.L\KernelType\().ProcessFilterCount1:
        ProcessFilterCountN SconvKernelFrame, \KernelType\(), 1

//
// Restore non-volatile registers and return.
//

.L\KernelType\().ExitKernel:
        ld.d    $a1, $sp, Filter_save_offset       //restore  Filter
        ld.d    $s0, $sp, 0*8
        ld.d    $s1, $sp, 1*8
        ld.d    $s2, $sp, 2*8
        ld.d    $s3, $sp, 3*8
        ld.d    $s4, $sp, 4*8
        ld.d    $ra, $sp, 5*8

        addi.d  $sp, $sp, SP_SIZE
        jr $ra
.endm

/*++

Macro Description:

    This macro generates code for the inner convolution kernel for the special
    case of a depthwise separable convolution.

Arguments:

    BlockSize - Supplies the number of elements per block.

    Isa - Supplies the instruction set architecture string for function tags.

--*/

        .macro SconvKernelDepthwiseFunction BlockSize, Isa

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

    Depthwise separable convolutions are a form of grouped convolution where
    the number of input and output channels per group are one.

Arguments:

    Input a0 - Supplies the address of the input buffer.

        The address is biased to include padding blocks for the left width
        dimension. The address is not biased to include padding rows for the
        left height dimension  these are accounted for in the outer kernel.

    Filter a1 - Supplies the address of the filter buffer.

    Output a2 - Supplies the address of the output buffer.

    StrideWidth a3 - Supplies the length in bytes of the blocked stride width.

    DilationWidth a4 - Supplies the length in bytes of the blocked dilation
        width.

    InputStride a5 - Supplies the length in bytes to advance the input buffer
        to the next input row.

    KernelHeight a6 - Supplies the height of the kernel to apply. This height may
        be less than the original kernel height after removing any padding
        rows.

    KernelWidth a7- Supplies the width of the kernel to apply.

    InputBase (sp, 0*8)- Supplies the address of the valid input buffer.

        This parameter is similar to the Input parameter, but does not include
        the padding blocks for the left width dimension. This parameter is used
        with the following InputWidth parameter in order to validate that the
        current input buffer address in bounds and not in the left or right
        width padding region.

    InputWidth (sp, 1*8)- Supplies the length in bytes of the blocked input width.

    DilatedInputWidth (sp, 2*8)- Supplies the length in bytes to advance the input base
        buffer to the next input row including dilation.

    OutputCountLeftPad (sp, 3*8)- Supplies the number of output elements that include
        one or more padding elements from the left edge.

    OutputCount (sp, 4*8)- Supplies the number of output elements that do not include
        any padding elements.

    OutputCountRightPad (sp, 5*8)- Supplies the number of output elements that include
        one or more padding elements from the right edge.

    Bias (sp, 6*8)- Supplies the address of the bias buffer.

    Flags (sp, 7*8)- Supplies additional flags controlling the convolution operation,
        especially post calculation options.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConvDepthwiseFloatKernel\Isa\()
        addi.d  $sp, $sp, -SP_SIZE
        st.d    $s0, $sp, 0*8
        st.d    $s1, $sp, 1*8
        st.d    $s2, $sp, 2*8
        st.d    $s3, $sp, 3*8
        st.d    $s4, $sp, 4*8
        st.d    $ra, $sp, 5*8

        st.d    $a6, $sp, KernelHeight_arg
        st.d    $a7, $sp, KernelWidth_arg

        ld.d    $s0, $sp, SP_SIZE+0*8
        ld.d    $s1, $sp, SP_SIZE+1*8
        ld.d    $s2, $sp, SP_SIZE+2*8
        ld.d    $s3, $sp, SP_SIZE+3*8
        st.d    $s0, $sp, InputBase_arg
        st.d    $s1, $sp, InputWidth_arg
        st.d    $s2, $sp, DilatedInputWidth_arg
        st.d    $s3, $sp, OutputCountLeftPad_arg
        ld.d    $s0, $sp, SP_SIZE+4*8
        ld.d    $s1, $sp, SP_SIZE+5*8
        ld.d    $s2, $sp, SP_SIZE+6*8
        ld.d    $s3, $sp, SP_SIZE+7*8
        st.d    $s0, $sp, OutputCount_arg
        st.d    $s1, $sp, OutputCountRightPad_arg
        st.d    $s2, $sp, Bias_arg
        st.d    $s3, $sp, Flags_arg
//
// Process the specified number of filter rows.
//
        move    $t8, $a4        // shuffle to Win64 register usage
        move    $t5, $a5
        move    $a4, $a2
        move    $a5, $a3
        ProcessFilterCountN SconvKernelDepthwiseFrame, Depthwise, 1

//
// Restore non-volatile registers and return.
        ld.d    $s0, $sp, 0*8
        ld.d    $s1, $sp, 1*8
        ld.d    $s2, $sp, 2*8
        ld.d    $s3, $sp, 3*8
        ld.d    $s4, $sp, 4*8
        ld.d    $ra, $sp, 5*8
        addi.d  $sp, $sp, SP_SIZE
//
        jr $ra
.endm

/*++

Macro Description:

    This macro generates code to compute the convolution for a vector of input
    blocks and a vector of filter blocks to produce a matrix of output blocks
    for a pointwise convolution.

Arguments:

    Isa - Supplies the instruction set architecture string for function tags.

    BlockSize - Supplies the number of elements per block.

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    (a0) - Supplies the address of the input buffer.

    (a1) - Supplies the FilterStride parameter (see function description).

    (s8) - Supplies the InputStride parameter (see function description).

    (a4) - Supplies the address of the output buffer.

    (a5) - Supplies the StrideWidth parameter (see function description).

    (s5) - Supplies the address of the filter buffer.

--*/

        .macro ProcessPointwiseOutputCountN Isa, BlockSize, FilterCount, OutputCount

        move    $a3, $a0
        move    $a2, $t2
        ld.d    $t1, $sp, InputChannels_arg
        ClearBlock \FilterCount\(), \OutputCount\()

.LPointwise.\FilterCount\().\OutputCount\().ProcessNextInputBlock:
.if \OutputCount\() > 3
        li.d    $s0, 2
        mul     $s0, $s0, $a5
        add.d   $t4, $a5, $s0
        add.d   $t4, $t4, $a3               # compute input plus 3 blocks
.endif
.if \FilterCount\() > 2
        li.d    $s0, 2             # compute filter plus 2 rows
        mul.d   $s0, $s0, $a1
        add.d   $t7, $a2, $s0
.endif

.if \BlockSize\() == 16
        .irp Index, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            ComputeBlock Pointwise, \FilterCount\(), \OutputCount\(), \Index\()*16*4, \Index\()*4
        .endr
.else
        .irp Index, 0, 1, 2, 3, 4, 5, 6, 7
            ComputeBlock Pointwise, \FilterCount\(), \OutputCount\(), (\Index\()-4)*8*4, \Index\()*4
        .endr
.endif
        add.d   $a3, $a3, $t8                     # advance input to next channel block
        addi.d  $a2, $a2, \BlockSize\()*\BlockSize\()*4
                                            # advance filter by 8i8o/16i16o block
        addi.d  $t1, $t1, -1               //InputChannels  decrement input blocks remaining
        bnez    $t1, .LPointwise.\FilterCount\().\OutputCount\().ProcessNextInputBlock

//
// Handle post processing of the output block.
//
        ld.w    $a2, $sp, Flags_arg     #load flag
.if \FilterCount\() > 1
        ld.d    $t6 ,$sp, OutputStride_arg        #load .LSconvKernelPointwiseFrame_OutputStride
.endif
        ld.d    $a3, $sp, Bias_arg        # load .LSconvKernelPointwiseFrame_Bias
        bl  MlasConvPostProcessFloat\Isa\()Filter\FilterCount\()Output\OutputCount\()
.endm

        .macro SconvKernelPointwiseFunction Isa, BiasFilter

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

    Pointwise convolutions have a kernel size of one. To simplify this
    implementation, no input padding is allowed, which matches typical usage in
    models.

Arguments:

    Input (a0) - Supplies the address of the input buffer.

    Filter (a1) - Supplies the address of the filter buffer.

    Output (a2) - Supplies the address of the output buffer.

    StrideWidth (a3) - Supplies the length in bytes of the blocked stride width.

    InputChannels (a4) - Supplies the number of input channels to process.

    FilterCount (a5) - Supplies the number of rows from the filter to process.

    InputStride (a6) - Supplies the length in bytes to advance the input buffer to
        the next input channel of the same input row.

    FilterStride (a7) - Supplies the length in bytes to advance the filter buffer
        to the next set of filters.

    OutputStride (sp+0) - Supplies the length in bytes to advance the output buffer
        to the next output address associated with the next set of filters.

    OutputCount (sp+8) - Supplies the number of output elements.

    Bias (sp+16) - Supplies the address of the bias buffer.

    Flags (sp+24) - Supplies additional flags controlling the convolution operation,
        especially post calculation options.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConvPointwiseFloatKernel\Isa\()
        addi.d  $sp, $sp, -SP_SIZE
        st.d    $s0, $sp, 0*8
        st.d    $s1, $sp, 1*8
        st.d    $s2, $sp, 2*8
        st.d    $s3, $sp, 3*8
        st.d    $s4, $sp, 4*8
        st.d    $ra, $sp, 5*8

        ld.d    $s0, $sp, SP_SIZE+0*8
        ld.d    $s1, $sp, SP_SIZE+1*8
        ld.d    $s2, $sp, SP_SIZE+2*8
        ld.d    $s3, $sp, SP_SIZE+3*8
        st.d    $s0, $sp, OutputStride_arg
        st.d    $s1, $sp, OutputCount_arg
        st.d    $s2, $sp, Bias_arg
        st.d    $s3, $sp, Flags_arg
        st.d    $a4, $sp, InputChannels_arg

.ifeqs "\BiasFilter\()","BiasFilter"
        addi.d    $t2, $a1, 4*8*4
.else
        move     $t2, $a1
.endif

        ld.d    $t0, $sp, OutputCount_arg      //OutputCount
        move    $a1, $a7        // FilterStride
        move    $t8, $a6        // InputStride
        move    $t1, $a5        // shuffle to Win64 register usage
        move    $a4, $a2
        move    $a5, $a3

//
// Process the specified number of filter rows.
//
        li.d    $s0, 3
        beq     $t1, $s0, .LPointwise.ProcessFilterCount3
        blt     $t1, $s0, .LPointwise.ProcessFilterCountLessThan3
        ProcessPointwiseFilterCountN 4
        b       .LPointwise.ExitKernel

.LPointwise.ProcessFilterCount3:
        ProcessPointwiseFilterCountN 3
        b     .LPointwise.ExitKernel

.LPointwise.ProcessFilterCountLessThan3:
        li.d    $s0, 2
        blt     $t1, $s0, .LPointwise.ProcessFilterCount1
        ProcessPointwiseFilterCountN 2
        b       .LPointwise.ExitKernel

.LPointwise.ProcessFilterCount1:
        ProcessPointwiseFilterCountN 1

//
// Restore non-volatile registers and return.
//
.LPointwise.ExitKernel:

        ld.d    $s0, $sp, 0*8
        ld.d    $s1, $sp, 1*8
        ld.d    $s2, $sp, 2*8
        ld.d    $s3, $sp, 3*8
        ld.d    $s4, $sp, 4*8
        ld.d    $ra, $sp, 5*8
        addi.d  $sp, $sp, SP_SIZE
        jr $ra
.endm
