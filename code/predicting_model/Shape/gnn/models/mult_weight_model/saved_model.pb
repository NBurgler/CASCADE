ʵ$
�3�3
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint�
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( ""
Ttype:
2	"
Tidxtype0:
2	
$
DisableCopyOnRead
resource�
9
DivNoNan
x"T
y"T
z"T"
Ttype:

2
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
:
OnesLike
x"T
y"T"
Ttype:
2	

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
�
StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
�
TopKV2

input"T
k"Tk
values"T
indices"
index_type"
sortedbool("
Ttype:
2	"
Tktype0:
2	"

index_typetype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
�
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T""
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.12v2.15.0-11-g63f5a65c7cd8��#
�
gru/gru_cell/biasVarHandleOp*
_output_shapes
: *"

debug_namegru/gru_cell/bias/*
dtype0*
shape
:*"
shared_namegru/gru_cell/bias
w
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
_output_shapes

:*
dtype0
�
gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *.

debug_name gru/gru_cell/recurrent_kernel/*
dtype0*
shape
:*.
shared_namegru/gru_cell/recurrent_kernel
�
1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel*
_output_shapes

:*
dtype0
�
gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *$

debug_namegru/gru_cell/kernel/*
dtype0*
shape:	�*$
shared_namegru/gru_cell/kernel
|
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes
:	�*
dtype0
�
dense_22/biasVarHandleOp*
_output_shapes
: *

debug_namedense_22/bias/*
dtype0*
shape:�*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:�*
dtype0
�
dense_22/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_22/kernel/*
dtype0*
shape:
��* 
shared_namedense_22/kernel
u
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel* 
_output_shapes
:
��*
dtype0
�
dense_21/biasVarHandleOp*
_output_shapes
: *

debug_namedense_21/bias/*
dtype0*
shape:�*
shared_namedense_21/bias
l
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes	
:�*
dtype0
�
dense_21/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_21/kernel/*
dtype0*
shape:
��* 
shared_namedense_21/kernel
u
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel* 
_output_shapes
:
��*
dtype0
�
dense_20/biasVarHandleOp*
_output_shapes
: *

debug_namedense_20/bias/*
dtype0*
shape:�*
shared_namedense_20/bias
l
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes	
:�*
dtype0
�
dense_20/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_20/kernel/*
dtype0*
shape:
��* 
shared_namedense_20/kernel
u
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel* 
_output_shapes
:
��*
dtype0
�
dense_19/biasVarHandleOp*
_output_shapes
: *

debug_namedense_19/bias/*
dtype0*
shape:�*
shared_namedense_19/bias
l
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes	
:�*
dtype0
�
dense_19/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_19/kernel/*
dtype0*
shape:
��* 
shared_namedense_19/kernel
u
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel* 
_output_shapes
:
��*
dtype0
�
dense_18/biasVarHandleOp*
_output_shapes
: *

debug_namedense_18/bias/*
dtype0*
shape:�*
shared_namedense_18/bias
l
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes	
:�*
dtype0
�
dense_18/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_18/kernel/*
dtype0*
shape:
��* 
shared_namedense_18/kernel
u
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel* 
_output_shapes
:
��*
dtype0
�
dense_17/biasVarHandleOp*
_output_shapes
: *

debug_namedense_17/bias/*
dtype0*
shape:�*
shared_namedense_17/bias
l
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes	
:�*
dtype0
�
dense_17/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_17/kernel/*
dtype0*
shape:
��* 
shared_namedense_17/kernel
u
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel* 
_output_shapes
:
��*
dtype0
�
dense_16/biasVarHandleOp*
_output_shapes
: *

debug_namedense_16/bias/*
dtype0*
shape:�*
shared_namedense_16/bias
l
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes	
:�*
dtype0
�
dense_16/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_16/kernel/*
dtype0*
shape:
��* 
shared_namedense_16/kernel
u
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel* 
_output_shapes
:
��*
dtype0
�
dense_15/biasVarHandleOp*
_output_shapes
: *

debug_namedense_15/bias/*
dtype0*
shape:�*
shared_namedense_15/bias
l
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes	
:�*
dtype0
�
dense_15/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_15/kernel/*
dtype0*
shape:
��* 
shared_namedense_15/kernel
u
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel* 
_output_shapes
:
��*
dtype0
�
dense_14/biasVarHandleOp*
_output_shapes
: *

debug_namedense_14/bias/*
dtype0*
shape:�*
shared_namedense_14/bias
l
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes	
:�*
dtype0
�
dense_14/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_14/kernel/*
dtype0*
shape:
��* 
shared_namedense_14/kernel
u
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel* 
_output_shapes
:
��*
dtype0
�
dense_13/biasVarHandleOp*
_output_shapes
: *

debug_namedense_13/bias/*
dtype0*
shape:�*
shared_namedense_13/bias
l
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes	
:�*
dtype0
�
dense_13/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_13/kernel/*
dtype0*
shape:
��* 
shared_namedense_13/kernel
u
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel* 
_output_shapes
:
��*
dtype0
�
dense_12/biasVarHandleOp*
_output_shapes
: *

debug_namedense_12/bias/*
dtype0*
shape:�*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:�*
dtype0
�
dense_12/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_12/kernel/*
dtype0*
shape:
��* 
shared_namedense_12/kernel
u
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel* 
_output_shapes
:
��*
dtype0
�
dense_11/biasVarHandleOp*
_output_shapes
: *

debug_namedense_11/bias/*
dtype0*
shape:�*
shared_namedense_11/bias
l
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes	
:�*
dtype0
�
dense_11/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_11/kernel/*
dtype0*
shape:
��* 
shared_namedense_11/kernel
u
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel* 
_output_shapes
:
��*
dtype0
�
dense_10/biasVarHandleOp*
_output_shapes
: *

debug_namedense_10/bias/*
dtype0*
shape:�*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:�*
dtype0
�
dense_10/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_10/kernel/*
dtype0*
shape:
��* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
��*
dtype0
�
dense_9/biasVarHandleOp*
_output_shapes
: *

debug_namedense_9/bias/*
dtype0*
shape:�*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes	
:�*
dtype0
�
dense_9/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_9/kernel/*
dtype0*
shape:
��*
shared_namedense_9/kernel
s
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel* 
_output_shapes
:
��*
dtype0
�
dense_8/biasVarHandleOp*
_output_shapes
: *

debug_namedense_8/bias/*
dtype0*
shape:�*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:�*
dtype0
�
dense_8/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_8/kernel/*
dtype0*
shape:
��*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
��*
dtype0
�
dense_7/biasVarHandleOp*
_output_shapes
: *

debug_namedense_7/bias/*
dtype0*
shape:�*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:�*
dtype0
�
dense_7/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_7/kernel/*
dtype0*
shape:
��*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
��*
dtype0
�
dense_6/biasVarHandleOp*
_output_shapes
: *

debug_namedense_6/bias/*
dtype0*
shape:�*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:�*
dtype0
�
dense_6/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_6/kernel/*
dtype0*
shape:
��*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
��*
dtype0
�
dense_5/biasVarHandleOp*
_output_shapes
: *

debug_namedense_5/bias/*
dtype0*
shape:�*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:�*
dtype0
�
dense_5/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_5/kernel/*
dtype0*
shape:
��*
shared_namedense_5/kernel
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel* 
_output_shapes
:
��*
dtype0
�
dense_4/biasVarHandleOp*
_output_shapes
: *

debug_namedense_4/bias/*
dtype0*
shape:�*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:�*
dtype0
�
dense_4/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_4/kernel/*
dtype0*
shape:
��*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
��*
dtype0
�
dense_3/biasVarHandleOp*
_output_shapes
: *

debug_namedense_3/bias/*
dtype0*
shape:�*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:�*
dtype0
�
dense_3/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_3/kernel/*
dtype0*
shape:
��*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
��*
dtype0
�
dense_2/biasVarHandleOp*
_output_shapes
: *

debug_namedense_2/bias/*
dtype0*
shape:�*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:�*
dtype0
�
dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape:
��*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
��*
dtype0
�
dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
�
dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
�

dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
�
dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape:
��*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��*
dtype0
�
edge_embedding/biasVarHandleOp*
_output_shapes
: *$

debug_nameedge_embedding/bias/*
dtype0*
shape:�*$
shared_nameedge_embedding/bias
x
'edge_embedding/bias/Read/ReadVariableOpReadVariableOpedge_embedding/bias*
_output_shapes	
:�*
dtype0
�
edge_embedding/kernelVarHandleOp*
_output_shapes
: *&

debug_nameedge_embedding/kernel/*
dtype0*
shape:	�*&
shared_nameedge_embedding/kernel
�
)edge_embedding/kernel/Read/ReadVariableOpReadVariableOpedge_embedding/kernel*
_output_shapes
:	�*
dtype0
�
stereo_embedding/biasVarHandleOp*
_output_shapes
: *&

debug_namestereo_embedding/bias/*
dtype0*
shape:*&
shared_namestereo_embedding/bias
{
)stereo_embedding/bias/Read/ReadVariableOpReadVariableOpstereo_embedding/bias*
_output_shapes
:*
dtype0
�
stereo_embedding/kernelVarHandleOp*
_output_shapes
: *(

debug_namestereo_embedding/kernel/*
dtype0*
shape
:*(
shared_namestereo_embedding/kernel
�
+stereo_embedding/kernel/Read/ReadVariableOpReadVariableOpstereo_embedding/kernel*
_output_shapes

:*
dtype0
�
"is_conjugated_embedding/embeddingsVarHandleOp*
_output_shapes
: *3

debug_name%#is_conjugated_embedding/embeddings/*
dtype0*
shape
:*3
shared_name$"is_conjugated_embedding/embeddings
�
6is_conjugated_embedding/embeddings/Read/ReadVariableOpReadVariableOp"is_conjugated_embedding/embeddings*
_output_shapes

:*
dtype0
�
bond_type_embedding/biasVarHandleOp*
_output_shapes
: *)

debug_namebond_type_embedding/bias/*
dtype0*
shape:*)
shared_namebond_type_embedding/bias
�
,bond_type_embedding/bias/Read/ReadVariableOpReadVariableOpbond_type_embedding/bias*
_output_shapes
:*
dtype0
�
bond_type_embedding/kernelVarHandleOp*
_output_shapes
: *+

debug_namebond_type_embedding/kernel/*
dtype0*
shape
:*+
shared_namebond_type_embedding/kernel
�
.bond_type_embedding/kernel/Read/ReadVariableOpReadVariableOpbond_type_embedding/kernel*
_output_shapes

:*
dtype0
�
node_embedding/biasVarHandleOp*
_output_shapes
: *$

debug_namenode_embedding/bias/*
dtype0*
shape:�*$
shared_namenode_embedding/bias
x
'node_embedding/bias/Read/ReadVariableOpReadVariableOpnode_embedding/bias*
_output_shapes	
:�*
dtype0
�
node_embedding/kernelVarHandleOp*
_output_shapes
: *&

debug_namenode_embedding/kernel/*
dtype0*
shape:	�*&
shared_namenode_embedding/kernel
�
)node_embedding/kernel/Read/ReadVariableOpReadVariableOpnode_embedding/kernel*
_output_shapes
:	�*
dtype0
�
valence_embedding/embeddingsVarHandleOp*
_output_shapes
: *-

debug_namevalence_embedding/embeddings/*
dtype0*
shape
:*-
shared_namevalence_embedding/embeddings
�
0valence_embedding/embeddings/Read/ReadVariableOpReadVariableOpvalence_embedding/embeddings*
_output_shapes

:*
dtype0
�
num_Hs_embedding/embeddingsVarHandleOp*
_output_shapes
: *,

debug_namenum_Hs_embedding/embeddings/*
dtype0*
shape
:*,
shared_namenum_Hs_embedding/embeddings
�
/num_Hs_embedding/embeddings/Read/ReadVariableOpReadVariableOpnum_Hs_embedding/embeddings*
_output_shapes

:*
dtype0
�
 no_implicit_embedding/embeddingsVarHandleOp*
_output_shapes
: *1

debug_name#!no_implicit_embedding/embeddings/*
dtype0*
shape
:*1
shared_name" no_implicit_embedding/embeddings
�
4no_implicit_embedding/embeddings/Read/ReadVariableOpReadVariableOp no_implicit_embedding/embeddings*
_output_shapes

:*
dtype0
�
 is_aromatic_embedding/embeddingsVarHandleOp*
_output_shapes
: *1

debug_name#!is_aromatic_embedding/embeddings/*
dtype0*
shape
:*1
shared_name" is_aromatic_embedding/embeddings
�
4is_aromatic_embedding/embeddings/Read/ReadVariableOpReadVariableOp is_aromatic_embedding/embeddings*
_output_shapes

:*
dtype0
�
"formal_charge_embedding/embeddingsVarHandleOp*
_output_shapes
: *3

debug_name%#formal_charge_embedding/embeddings/*
dtype0*
shape
:*3
shared_name$"formal_charge_embedding/embeddings
�
6formal_charge_embedding/embeddings/Read/ReadVariableOpReadVariableOp"formal_charge_embedding/embeddings*
_output_shapes

:*
dtype0
�
degree_embedding/embeddingsVarHandleOp*
_output_shapes
: *,

debug_namedegree_embedding/embeddings/*
dtype0*
shape
:*,
shared_namedegree_embedding/embeddings
�
/degree_embedding/embeddings/Read/ReadVariableOpReadVariableOpdegree_embedding/embeddings*
_output_shapes

:*
dtype0
�
hybridization_embedding/biasVarHandleOp*
_output_shapes
: *-

debug_namehybridization_embedding/bias/*
dtype0*
shape:*-
shared_namehybridization_embedding/bias
�
0hybridization_embedding/bias/Read/ReadVariableOpReadVariableOphybridization_embedding/bias*
_output_shapes
:*
dtype0
�
hybridization_embedding/kernelVarHandleOp*
_output_shapes
: */

debug_name!hybridization_embedding/kernel/*
dtype0*
shape
:	*/
shared_name hybridization_embedding/kernel
�
2hybridization_embedding/kernel/Read/ReadVariableOpReadVariableOphybridization_embedding/kernel*
_output_shapes

:	*
dtype0
�
chiral_tag_embedding/biasVarHandleOp*
_output_shapes
: **

debug_namechiral_tag_embedding/bias/*
dtype0*
shape:**
shared_namechiral_tag_embedding/bias
�
-chiral_tag_embedding/bias/Read/ReadVariableOpReadVariableOpchiral_tag_embedding/bias*
_output_shapes
:*
dtype0
�
chiral_tag_embedding/kernelVarHandleOp*
_output_shapes
: *,

debug_namechiral_tag_embedding/kernel/*
dtype0*
shape
:	*,
shared_namechiral_tag_embedding/kernel
�
/chiral_tag_embedding/kernel/Read/ReadVariableOpReadVariableOpchiral_tag_embedding/kernel*
_output_shapes

:	*
dtype0
�
atom_sym_embedding/biasVarHandleOp*
_output_shapes
: *(

debug_nameatom_sym_embedding/bias/*
dtype0*
shape:*(
shared_nameatom_sym_embedding/bias

+atom_sym_embedding/bias/Read/ReadVariableOpReadVariableOpatom_sym_embedding/bias*
_output_shapes
:*
dtype0
�
atom_sym_embedding/kernelVarHandleOp*
_output_shapes
: **

debug_nameatom_sym_embedding/kernel/*
dtype0*
shape
:**
shared_nameatom_sym_embedding/kernel
�
-atom_sym_embedding/kernel/Read/ReadVariableOpReadVariableOpatom_sym_embedding/kernel*
_output_shapes

:*
dtype0
i
serve_examplesPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserve_examplesatom_sym_embedding/kernelatom_sym_embedding/biaschiral_tag_embedding/kernelchiral_tag_embedding/biashybridization_embedding/kernelhybridization_embedding/biasdegree_embedding/embeddings"formal_charge_embedding/embeddings is_aromatic_embedding/embeddings no_implicit_embedding/embeddingsnum_Hs_embedding/embeddingsvalence_embedding/embeddingsnode_embedding/kernelnode_embedding/biasbond_type_embedding/kernelbond_type_embedding/bias"is_conjugated_embedding/embeddingsstereo_embedding/kernelstereo_embedding/biasedge_embedding/kerneledge_embedding/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/bias*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*h
_read_only_resource_inputsJ
HF	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEF*-
config_proto

CPU

GPU 2J 8� *6
f1R/
-__inference_signature_wrapper___call___316305
s
serving_default_examplesPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_examplesatom_sym_embedding/kernelatom_sym_embedding/biaschiral_tag_embedding/kernelchiral_tag_embedding/biashybridization_embedding/kernelhybridization_embedding/biasdegree_embedding/embeddings"formal_charge_embedding/embeddings is_aromatic_embedding/embeddings no_implicit_embedding/embeddingsnum_Hs_embedding/embeddingsvalence_embedding/embeddingsnode_embedding/kernelnode_embedding/biasbond_type_embedding/kernelbond_type_embedding/bias"is_conjugated_embedding/embeddingsstereo_embedding/kernelstereo_embedding/biasedge_embedding/kerneledge_embedding/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/bias*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*h
_read_only_resource_inputsJ
HF	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEF*-
config_proto

CPU

GPU 2J 8� *6
f1R/
-__inference_signature_wrapper___call___316450

NoOpNoOp
�>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�=
value�=B�= B�=
�
_endpoint_names
_endpoint_signatures
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve
	
signatures*
* 

	
serve* 
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
 21
!22
"23
#24
$25
%26
&27
'28
(29
)30
*31
+32
,33
-34
.35
/36
037
138
239
340
441
542
643
744
845
946
:47
;48
<49
=50
>51
?52
@53
A54
B55
C56
D57
E58
F59
G60
H61
I62
J63
K64
L65
M66
N67
O68
P69*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
 21
!22
"23
#24
$25
%26
&27
'28
(29
)30
*31
+32
,33
-34
.35
/36
037
138
239
340
441
542
643
744
845
946
:47
;48
<49
=50
>51
?52
@53
A54
B55
C56
D57
E58
F59
G60
H61
I62
J63
K64
L65
M66
N67
O68
P69*
* 
�
0
1
2
3
14
65
96
E7
N8
9
10
#11
)12
-13
714
;15
B16
I17
O18
19
 20
!21
$22
/23
324
C25
K26
P27
28
29
30
31
&32
833
<34
35
36
%37
=38
@39
H40
J41
L42
43
044
245
:46
>47
G48
M49
50
51
52
"53
'54
(55
,56
457
?58
F59
60
61
62
63
*64
+65
.66
567
A68
D69*
* 

Qtrace_0* 
"
	Rserve
Sserving_default* 
* 
YS
VARIABLE_VALUEatom_sym_embedding/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEatom_sym_embedding/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEchiral_tag_embedding/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEchiral_tag_embedding/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEhybridization_embedding/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEhybridization_embedding/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdegree_embedding/embeddings&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"formal_charge_embedding/embeddings&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE is_aromatic_embedding/embeddings&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE no_implicit_embedding/embeddings&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEnum_Hs_embedding/embeddings'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEvalence_embedding/embeddings'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEnode_embedding/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEnode_embedding/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbond_type_embedding/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbond_type_embedding/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"is_conjugated_embedding/embeddings'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEstereo_embedding/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEstereo_embedding/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEedge_embedding/kernel'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEedge_embedding/bias'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
dense/bias'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_1/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_1/bias'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_2/kernel'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_2/bias'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_3/kernel'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_3/bias'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_4/kernel'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_4/bias'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_5/kernel'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_5/bias'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_6/kernel'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_6/bias'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_7/kernel'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_7/bias'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_8/kernel'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_8/bias'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_9/kernel'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_9/bias'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_10/kernel'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_10/bias'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_11/kernel'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_11/bias'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_12/kernel'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_12/bias'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_13/kernel'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_13/bias'variables/48/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_14/kernel'variables/49/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_14/bias'variables/50/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_15/kernel'variables/51/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_15/bias'variables/52/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_16/kernel'variables/53/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_16/bias'variables/54/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_17/kernel'variables/55/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_17/bias'variables/56/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_18/kernel'variables/57/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_18/bias'variables/58/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_19/kernel'variables/59/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_19/bias'variables/60/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_20/kernel'variables/61/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_20/bias'variables/62/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_21/kernel'variables/63/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_21/bias'variables/64/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_22/kernel'variables/65/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_22/bias'variables/66/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEgru/gru_cell/kernel'variables/67/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEgru/gru_cell/recurrent_kernel'variables/68/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEgru/gru_cell/bias'variables/69/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameatom_sym_embedding/kernelatom_sym_embedding/biaschiral_tag_embedding/kernelchiral_tag_embedding/biashybridization_embedding/kernelhybridization_embedding/biasdegree_embedding/embeddings"formal_charge_embedding/embeddings is_aromatic_embedding/embeddings no_implicit_embedding/embeddingsnum_Hs_embedding/embeddingsvalence_embedding/embeddingsnode_embedding/kernelnode_embedding/biasbond_type_embedding/kernelbond_type_embedding/bias"is_conjugated_embedding/embeddingsstereo_embedding/kernelstereo_embedding/biasedge_embedding/kerneledge_embedding/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biasConst*S
TinL
J2H*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_316894
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameatom_sym_embedding/kernelatom_sym_embedding/biaschiral_tag_embedding/kernelchiral_tag_embedding/biashybridization_embedding/kernelhybridization_embedding/biasdegree_embedding/embeddings"formal_charge_embedding/embeddings is_aromatic_embedding/embeddings no_implicit_embedding/embeddingsnum_Hs_embedding/embeddingsvalence_embedding/embeddingsnode_embedding/kernelnode_embedding/biasbond_type_embedding/kernelbond_type_embedding/bias"is_conjugated_embedding/embeddingsstereo_embedding/kernelstereo_embedding/biasedge_embedding/kerneledge_embedding/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/bias*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_317113��!
�4
�
(__inference_gpu_gru_with_fallback_316019

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:���������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	�:	�:	�*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:::*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������U
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes
:0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	�[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	�[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	�[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:Z
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:Z
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes
:@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:Z
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:�1U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:���������:���������: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:���������p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:���������*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:���������]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:���������Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:���������I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:����������:���������:	�::*<
api_implements*(gru_95a99024-6e46-4256-a2c3-70de478e526d*
api_preferred_deviceGPU*
go_backwards( *

time_major( :D@

_output_shapes

:

_user_specified_namebias:PL

_output_shapes

:
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	�
 
_user_specified_namekernel:OK
'
_output_shapes
:���������
 
_user_specified_nameinit_h:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�+
"__inference__traced_restore_317113
file_prefix<
*assignvariableop_atom_sym_embedding_kernel:8
*assignvariableop_1_atom_sym_embedding_bias:@
.assignvariableop_2_chiral_tag_embedding_kernel:	:
,assignvariableop_3_chiral_tag_embedding_bias:C
1assignvariableop_4_hybridization_embedding_kernel:	=
/assignvariableop_5_hybridization_embedding_bias:@
.assignvariableop_6_degree_embedding_embeddings:G
5assignvariableop_7_formal_charge_embedding_embeddings:E
3assignvariableop_8_is_aromatic_embedding_embeddings:E
3assignvariableop_9_no_implicit_embedding_embeddings:A
/assignvariableop_10_num_hs_embedding_embeddings:B
0assignvariableop_11_valence_embedding_embeddings:<
)assignvariableop_12_node_embedding_kernel:	�6
'assignvariableop_13_node_embedding_bias:	�@
.assignvariableop_14_bond_type_embedding_kernel::
,assignvariableop_15_bond_type_embedding_bias:H
6assignvariableop_16_is_conjugated_embedding_embeddings:=
+assignvariableop_17_stereo_embedding_kernel:7
)assignvariableop_18_stereo_embedding_bias:<
)assignvariableop_19_edge_embedding_kernel:	�6
'assignvariableop_20_edge_embedding_bias:	�4
 assignvariableop_21_dense_kernel:
��-
assignvariableop_22_dense_bias:	�6
"assignvariableop_23_dense_1_kernel:
��/
 assignvariableop_24_dense_1_bias:	�6
"assignvariableop_25_dense_2_kernel:
��/
 assignvariableop_26_dense_2_bias:	�6
"assignvariableop_27_dense_3_kernel:
��/
 assignvariableop_28_dense_3_bias:	�6
"assignvariableop_29_dense_4_kernel:
��/
 assignvariableop_30_dense_4_bias:	�6
"assignvariableop_31_dense_5_kernel:
��/
 assignvariableop_32_dense_5_bias:	�6
"assignvariableop_33_dense_6_kernel:
��/
 assignvariableop_34_dense_6_bias:	�6
"assignvariableop_35_dense_7_kernel:
��/
 assignvariableop_36_dense_7_bias:	�6
"assignvariableop_37_dense_8_kernel:
��/
 assignvariableop_38_dense_8_bias:	�6
"assignvariableop_39_dense_9_kernel:
��/
 assignvariableop_40_dense_9_bias:	�7
#assignvariableop_41_dense_10_kernel:
��0
!assignvariableop_42_dense_10_bias:	�7
#assignvariableop_43_dense_11_kernel:
��0
!assignvariableop_44_dense_11_bias:	�7
#assignvariableop_45_dense_12_kernel:
��0
!assignvariableop_46_dense_12_bias:	�7
#assignvariableop_47_dense_13_kernel:
��0
!assignvariableop_48_dense_13_bias:	�7
#assignvariableop_49_dense_14_kernel:
��0
!assignvariableop_50_dense_14_bias:	�7
#assignvariableop_51_dense_15_kernel:
��0
!assignvariableop_52_dense_15_bias:	�7
#assignvariableop_53_dense_16_kernel:
��0
!assignvariableop_54_dense_16_bias:	�7
#assignvariableop_55_dense_17_kernel:
��0
!assignvariableop_56_dense_17_bias:	�7
#assignvariableop_57_dense_18_kernel:
��0
!assignvariableop_58_dense_18_bias:	�7
#assignvariableop_59_dense_19_kernel:
��0
!assignvariableop_60_dense_19_bias:	�7
#assignvariableop_61_dense_20_kernel:
��0
!assignvariableop_62_dense_20_bias:	�7
#assignvariableop_63_dense_21_kernel:
��0
!assignvariableop_64_dense_21_bias:	�7
#assignvariableop_65_dense_22_kernel:
��0
!assignvariableop_66_dense_22_bias:	�:
'assignvariableop_67_gru_gru_cell_kernel:	�C
1assignvariableop_68_gru_gru_cell_recurrent_kernel:7
%assignvariableop_69_gru_gru_cell_bias:
identity_71��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*�
value�B�GB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB'variables/62/.ATTRIBUTES/VARIABLE_VALUEB'variables/63/.ATTRIBUTES/VARIABLE_VALUEB'variables/64/.ATTRIBUTES/VARIABLE_VALUEB'variables/65/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/68/.ATTRIBUTES/VARIABLE_VALUEB'variables/69/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*�
value�B�GB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*U
dtypesK
I2G[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp*assignvariableop_atom_sym_embedding_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp*assignvariableop_1_atom_sym_embedding_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_chiral_tag_embedding_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp,assignvariableop_3_chiral_tag_embedding_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp1assignvariableop_4_hybridization_embedding_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp/assignvariableop_5_hybridization_embedding_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp.assignvariableop_6_degree_embedding_embeddingsIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp5assignvariableop_7_formal_charge_embedding_embeddingsIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp3assignvariableop_8_is_aromatic_embedding_embeddingsIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp3assignvariableop_9_no_implicit_embedding_embeddingsIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_num_hs_embedding_embeddingsIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp0assignvariableop_11_valence_embedding_embeddingsIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp)assignvariableop_12_node_embedding_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp'assignvariableop_13_node_embedding_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp.assignvariableop_14_bond_type_embedding_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp,assignvariableop_15_bond_type_embedding_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_is_conjugated_embedding_embeddingsIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp+assignvariableop_17_stereo_embedding_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_stereo_embedding_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_edge_embedding_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_edge_embedding_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_dense_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp"assignvariableop_23_dense_1_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp assignvariableop_24_dense_1_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_dense_2_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp assignvariableop_26_dense_2_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp"assignvariableop_27_dense_3_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp assignvariableop_28_dense_3_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp"assignvariableop_29_dense_4_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp assignvariableop_30_dense_4_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp"assignvariableop_31_dense_5_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp assignvariableop_32_dense_5_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp"assignvariableop_33_dense_6_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp assignvariableop_34_dense_6_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp"assignvariableop_35_dense_7_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp assignvariableop_36_dense_7_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp"assignvariableop_37_dense_8_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp assignvariableop_38_dense_8_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp"assignvariableop_39_dense_9_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp assignvariableop_40_dense_9_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp#assignvariableop_41_dense_10_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp!assignvariableop_42_dense_10_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp#assignvariableop_43_dense_11_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp!assignvariableop_44_dense_11_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp#assignvariableop_45_dense_12_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp!assignvariableop_46_dense_12_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp#assignvariableop_47_dense_13_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp!assignvariableop_48_dense_13_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp#assignvariableop_49_dense_14_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp!assignvariableop_50_dense_14_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp#assignvariableop_51_dense_15_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp!assignvariableop_52_dense_15_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp#assignvariableop_53_dense_16_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp!assignvariableop_54_dense_16_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp#assignvariableop_55_dense_17_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp!assignvariableop_56_dense_17_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp#assignvariableop_57_dense_18_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp!assignvariableop_58_dense_18_biasIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp#assignvariableop_59_dense_19_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp!assignvariableop_60_dense_19_biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp#assignvariableop_61_dense_20_kernelIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp!assignvariableop_62_dense_20_biasIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp#assignvariableop_63_dense_21_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp!assignvariableop_64_dense_21_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp#assignvariableop_65_dense_22_kernelIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp!assignvariableop_66_dense_22_biasIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp'assignvariableop_67_gru_gru_cell_kernelIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp1assignvariableop_68_gru_gru_cell_recurrent_kernelIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp%assignvariableop_69_gru_gru_cell_biasIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_70Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_71IdentityIdentity_70:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_71Identity_71:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:1F-
+
_user_specified_namegru/gru_cell/bias:=E9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3D/
-
_user_specified_namegru/gru_cell/kernel:-C)
'
_user_specified_namedense_22/bias:/B+
)
_user_specified_namedense_22/kernel:-A)
'
_user_specified_namedense_21/bias:/@+
)
_user_specified_namedense_21/kernel:-?)
'
_user_specified_namedense_20/bias:/>+
)
_user_specified_namedense_20/kernel:-=)
'
_user_specified_namedense_19/bias:/<+
)
_user_specified_namedense_19/kernel:-;)
'
_user_specified_namedense_18/bias:/:+
)
_user_specified_namedense_18/kernel:-9)
'
_user_specified_namedense_17/bias:/8+
)
_user_specified_namedense_17/kernel:-7)
'
_user_specified_namedense_16/bias:/6+
)
_user_specified_namedense_16/kernel:-5)
'
_user_specified_namedense_15/bias:/4+
)
_user_specified_namedense_15/kernel:-3)
'
_user_specified_namedense_14/bias:/2+
)
_user_specified_namedense_14/kernel:-1)
'
_user_specified_namedense_13/bias:/0+
)
_user_specified_namedense_13/kernel:-/)
'
_user_specified_namedense_12/bias:/.+
)
_user_specified_namedense_12/kernel:--)
'
_user_specified_namedense_11/bias:/,+
)
_user_specified_namedense_11/kernel:-+)
'
_user_specified_namedense_10/bias:/*+
)
_user_specified_namedense_10/kernel:,)(
&
_user_specified_namedense_9/bias:.(*
(
_user_specified_namedense_9/kernel:,'(
&
_user_specified_namedense_8/bias:.&*
(
_user_specified_namedense_8/kernel:,%(
&
_user_specified_namedense_7/bias:.$*
(
_user_specified_namedense_7/kernel:,#(
&
_user_specified_namedense_6/bias:."*
(
_user_specified_namedense_6/kernel:,!(
&
_user_specified_namedense_5/bias:. *
(
_user_specified_namedense_5/kernel:,(
&
_user_specified_namedense_4/bias:.*
(
_user_specified_namedense_4/kernel:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:3/
-
_user_specified_nameedge_embedding/bias:51
/
_user_specified_nameedge_embedding/kernel:51
/
_user_specified_namestereo_embedding/bias:73
1
_user_specified_namestereo_embedding/kernel:B>
<
_user_specified_name$"is_conjugated_embedding/embeddings:84
2
_user_specified_namebond_type_embedding/bias::6
4
_user_specified_namebond_type_embedding/kernel:3/
-
_user_specified_namenode_embedding/bias:51
/
_user_specified_namenode_embedding/kernel:<8
6
_user_specified_namevalence_embedding/embeddings:;7
5
_user_specified_namenum_Hs_embedding/embeddings:@
<
:
_user_specified_name" no_implicit_embedding/embeddings:@	<
:
_user_specified_name" is_aromatic_embedding/embeddings:B>
<
_user_specified_name$"formal_charge_embedding/embeddings:;7
5
_user_specified_namedegree_embedding/embeddings:<8
6
_user_specified_namehybridization_embedding/bias:>:
8
_user_specified_name hybridization_embedding/kernel:95
3
_user_specified_namechiral_tag_embedding/bias:;7
5
_user_specified_namechiral_tag_embedding/kernel:73
1
_user_specified_nameatom_sym_embedding/bias:95
3
_user_specified_nameatom_sym_embedding/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�4
�
-__inference_signature_wrapper___call___316450
examples
unknown:
	unknown_0:
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:	�

unknown_12:	�

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:	�

unknown_19:	�

unknown_20:
��

unknown_21:	�

unknown_22:
��

unknown_23:	�

unknown_24:
��

unknown_25:	�

unknown_26:
��

unknown_27:	�

unknown_28:
��

unknown_29:	�

unknown_30:
��

unknown_31:	�

unknown_32:
��

unknown_33:	�

unknown_34:
��

unknown_35:	�

unknown_36:
��

unknown_37:	�

unknown_38:
��

unknown_39:	�

unknown_40:
��

unknown_41:	�

unknown_42:
��

unknown_43:	�

unknown_44:
��

unknown_45:	�

unknown_46:
��

unknown_47:	�

unknown_48:
��

unknown_49:	�

unknown_50:
��

unknown_51:	�

unknown_52:
��

unknown_53:	�

unknown_54:
��

unknown_55:	�

unknown_56:
��

unknown_57:	�

unknown_58:
��

unknown_59:	�

unknown_60:
��

unknown_61:	�

unknown_62:
��

unknown_63:	�

unknown_64:
��

unknown_65:	�

unknown_66:	�

unknown_67:

unknown_68:
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*h
_read_only_resource_inputsJ
HF	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEF*-
config_proto

CPU

GPU 2J 8� *$
fR
__inference___call___316159s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&F"
 
_user_specified_name316446:&E"
 
_user_specified_name316444:&D"
 
_user_specified_name316442:&C"
 
_user_specified_name316440:&B"
 
_user_specified_name316438:&A"
 
_user_specified_name316436:&@"
 
_user_specified_name316434:&?"
 
_user_specified_name316432:&>"
 
_user_specified_name316430:&="
 
_user_specified_name316428:&<"
 
_user_specified_name316426:&;"
 
_user_specified_name316424:&:"
 
_user_specified_name316422:&9"
 
_user_specified_name316420:&8"
 
_user_specified_name316418:&7"
 
_user_specified_name316416:&6"
 
_user_specified_name316414:&5"
 
_user_specified_name316412:&4"
 
_user_specified_name316410:&3"
 
_user_specified_name316408:&2"
 
_user_specified_name316406:&1"
 
_user_specified_name316404:&0"
 
_user_specified_name316402:&/"
 
_user_specified_name316400:&."
 
_user_specified_name316398:&-"
 
_user_specified_name316396:&,"
 
_user_specified_name316394:&+"
 
_user_specified_name316392:&*"
 
_user_specified_name316390:&)"
 
_user_specified_name316388:&("
 
_user_specified_name316386:&'"
 
_user_specified_name316384:&&"
 
_user_specified_name316382:&%"
 
_user_specified_name316380:&$"
 
_user_specified_name316378:&#"
 
_user_specified_name316376:&""
 
_user_specified_name316374:&!"
 
_user_specified_name316372:& "
 
_user_specified_name316370:&"
 
_user_specified_name316368:&"
 
_user_specified_name316366:&"
 
_user_specified_name316364:&"
 
_user_specified_name316362:&"
 
_user_specified_name316360:&"
 
_user_specified_name316358:&"
 
_user_specified_name316356:&"
 
_user_specified_name316354:&"
 
_user_specified_name316352:&"
 
_user_specified_name316350:&"
 
_user_specified_name316348:&"
 
_user_specified_name316346:&"
 
_user_specified_name316344:&"
 
_user_specified_name316342:&"
 
_user_specified_name316340:&"
 
_user_specified_name316338:&"
 
_user_specified_name316336:&"
 
_user_specified_name316334:&"
 
_user_specified_name316332:&"
 
_user_specified_name316330:&"
 
_user_specified_name316328:&
"
 
_user_specified_name316326:&	"
 
_user_specified_name316324:&"
 
_user_specified_name316322:&"
 
_user_specified_name316320:&"
 
_user_specified_name316318:&"
 
_user_specified_name316316:&"
 
_user_specified_name316314:&"
 
_user_specified_name316312:&"
 
_user_specified_name316310:&"
 
_user_specified_name316308:M I
#
_output_shapes
:���������
"
_user_specified_name
examples
��
�
__inference___call___316159
examplesf
Tmodel_2_model_1_map_features_model_atom_sym_embedding_matmul_readvariableop_resource:c
Umodel_2_model_1_map_features_model_atom_sym_embedding_biasadd_readvariableop_resource:h
Vmodel_2_model_1_map_features_model_chiral_tag_embedding_matmul_readvariableop_resource:	e
Wmodel_2_model_1_map_features_model_chiral_tag_embedding_biasadd_readvariableop_resource:k
Ymodel_2_model_1_map_features_model_hybridization_embedding_matmul_readvariableop_resource:	h
Zmodel_2_model_1_map_features_model_hybridization_embedding_biasadd_readvariableop_resource:]
Kmodel_2_model_1_map_features_model_degree_embedding_embedding_lookup_315371:d
Rmodel_2_model_1_map_features_model_formal_charge_embedding_embedding_lookup_315375:b
Pmodel_2_model_1_map_features_model_is_aromatic_embedding_embedding_lookup_315379:b
Pmodel_2_model_1_map_features_model_no_implicit_embedding_embedding_lookup_315383:]
Kmodel_2_model_1_map_features_model_num_hs_embedding_embedding_lookup_315387:^
Lmodel_2_model_1_map_features_model_valence_embedding_embedding_lookup_315391:c
Pmodel_2_model_1_map_features_model_node_embedding_matmul_readvariableop_resource:	�`
Qmodel_2_model_1_map_features_model_node_embedding_biasadd_readvariableop_resource:	�i
Wmodel_2_model_1_map_features_model_1_bond_type_embedding_matmul_readvariableop_resource:f
Xmodel_2_model_1_map_features_model_1_bond_type_embedding_biasadd_readvariableop_resource:f
Tmodel_2_model_1_map_features_model_1_is_conjugated_embedding_embedding_lookup_315427:f
Tmodel_2_model_1_map_features_model_1_stereo_embedding_matmul_readvariableop_resource:c
Umodel_2_model_1_map_features_model_1_stereo_embedding_biasadd_readvariableop_resource:e
Rmodel_2_model_1_map_features_model_1_edge_embedding_matmul_readvariableop_resource:	�b
Smodel_2_model_1_map_features_model_1_edge_embedding_biasadd_readvariableop_resource:	��
}model_2_model_1_graph_update_edge_set_update_residual_next_state_sequential_3_sequential_dense_matmul_readvariableop_resource:
���
~model_2_model_1_graph_update_edge_set_update_residual_next_state_sequential_3_sequential_dense_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_edge_set_update_residual_next_state_sequential_3_sequential_1_dense_1_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_edge_set_update_residual_next_state_sequential_3_sequential_1_dense_1_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_edge_set_update_residual_next_state_sequential_3_sequential_2_dense_2_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_edge_set_update_residual_next_state_sequential_3_sequential_2_dense_2_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_4_dense_3_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_4_dense_3_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_5_dense_4_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_5_dense_4_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_6_dense_5_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_6_dense_5_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_7_dense_6_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_7_dense_6_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_1_edge_set_update_1_residual_next_state_2_sequential_12_sequential_9_dense_7_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_1_edge_set_update_1_residual_next_state_2_sequential_12_sequential_9_dense_7_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_1_edge_set_update_1_residual_next_state_2_sequential_12_sequential_10_dense_8_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_1_edge_set_update_1_residual_next_state_2_sequential_12_sequential_10_dense_8_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_1_edge_set_update_1_residual_next_state_2_sequential_12_sequential_11_dense_9_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_1_edge_set_update_1_residual_next_state_2_sequential_12_sequential_11_dense_9_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_13_dense_10_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_13_dense_10_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_14_dense_11_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_14_dense_11_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_15_dense_12_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_15_dense_12_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_16_dense_13_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_16_dense_13_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_2_edge_set_update_2_residual_next_state_4_sequential_21_sequential_18_dense_14_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_2_edge_set_update_2_residual_next_state_4_sequential_21_sequential_18_dense_14_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_2_edge_set_update_2_residual_next_state_4_sequential_21_sequential_19_dense_15_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_2_edge_set_update_2_residual_next_state_4_sequential_21_sequential_19_dense_15_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_2_edge_set_update_2_residual_next_state_4_sequential_21_sequential_20_dense_16_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_2_edge_set_update_2_residual_next_state_4_sequential_21_sequential_20_dense_16_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_22_dense_17_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_22_dense_17_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_23_dense_18_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_23_dense_18_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_24_dense_19_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_24_dense_19_biasadd_readvariableop_resource:	��
�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_25_dense_20_matmul_readvariableop_resource:
���
�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_25_dense_20_biasadd_readvariableop_resource:	�g
Smodel_2_model_1_sequential_29_sequential_27_dense_21_matmul_readvariableop_resource:
��c
Tmodel_2_model_1_sequential_29_sequential_27_dense_21_biasadd_readvariableop_resource:	�g
Smodel_2_model_1_sequential_29_sequential_28_dense_22_matmul_readvariableop_resource:
��c
Tmodel_2_model_1_sequential_29_sequential_28_dense_22_biasadd_readvariableop_resource:	�C
0model_2_model_1_gru_read_readvariableop_resource:	�D
2model_2_model_1_gru_read_1_readvariableop_resource:D
2model_2_model_1_gru_read_2_readvariableop_resource:
identity��umodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/BiasAdd/ReadVariableOp�tmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/MatMul/ReadVariableOp�ymodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/BiasAdd/ReadVariableOp�xmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/MatMul/ReadVariableOp�ymodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/BiasAdd/ReadVariableOp�xmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/MatMul/ReadVariableOp�{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/BiasAdd/ReadVariableOp�zmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/MatMul/ReadVariableOp�{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/BiasAdd/ReadVariableOp�zmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/MatMul/ReadVariableOp�{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/BiasAdd/ReadVariableOp�zmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/MatMul/ReadVariableOp�{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/BiasAdd/ReadVariableOp�zmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/MatMul/ReadVariableOp��model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/BiasAdd/ReadVariableOp��model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/MatMul/ReadVariableOp��model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/BiasAdd/ReadVariableOp��model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/MatMul/ReadVariableOp��model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/MatMul/ReadVariableOp��model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/BiasAdd/ReadVariableOp��model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/MatMul/ReadVariableOp��model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/BiasAdd/ReadVariableOp��model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/MatMul/ReadVariableOp��model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/BiasAdd/ReadVariableOp��model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/MatMul/ReadVariableOp��model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/BiasAdd/ReadVariableOp��model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/MatMul/ReadVariableOp��model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/BiasAdd/ReadVariableOp��model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/MatMul/ReadVariableOp��model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/BiasAdd/ReadVariableOp��model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/MatMul/ReadVariableOp��model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/BiasAdd/ReadVariableOp��model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/MatMul/ReadVariableOp��model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/BiasAdd/ReadVariableOp��model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/MatMul/ReadVariableOp��model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/BiasAdd/ReadVariableOp��model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/MatMul/ReadVariableOp��model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/BiasAdd/ReadVariableOp��model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/MatMul/ReadVariableOp��model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/BiasAdd/ReadVariableOp��model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/MatMul/ReadVariableOp�'model_2/model_1/gru/Read/ReadVariableOp�)model_2/model_1/gru/Read_1/ReadVariableOp�)model_2/model_1/gru/Read_2/ReadVariableOp�Lmodel_2/model_1/map_features/model/atom_sym_embedding/BiasAdd/ReadVariableOp�Kmodel_2/model_1/map_features/model/atom_sym_embedding/MatMul/ReadVariableOp�Nmodel_2/model_1/map_features/model/chiral_tag_embedding/BiasAdd/ReadVariableOp�Mmodel_2/model_1/map_features/model/chiral_tag_embedding/MatMul/ReadVariableOp�Dmodel_2/model_1/map_features/model/degree_embedding/embedding_lookup�Kmodel_2/model_1/map_features/model/formal_charge_embedding/embedding_lookup�Qmodel_2/model_1/map_features/model/hybridization_embedding/BiasAdd/ReadVariableOp�Pmodel_2/model_1/map_features/model/hybridization_embedding/MatMul/ReadVariableOp�Imodel_2/model_1/map_features/model/is_aromatic_embedding/embedding_lookup�Imodel_2/model_1/map_features/model/no_implicit_embedding/embedding_lookup�Hmodel_2/model_1/map_features/model/node_embedding/BiasAdd/ReadVariableOp�Gmodel_2/model_1/map_features/model/node_embedding/MatMul/ReadVariableOp�Dmodel_2/model_1/map_features/model/num_Hs_embedding/embedding_lookup�Emodel_2/model_1/map_features/model/valence_embedding/embedding_lookup�Omodel_2/model_1/map_features/model_1/bond_type_embedding/BiasAdd/ReadVariableOp�Nmodel_2/model_1/map_features/model_1/bond_type_embedding/MatMul/ReadVariableOp�Jmodel_2/model_1/map_features/model_1/edge_embedding/BiasAdd/ReadVariableOp�Imodel_2/model_1/map_features/model_1/edge_embedding/MatMul/ReadVariableOp�Mmodel_2/model_1/map_features/model_1/is_conjugated_embedding/embedding_lookup�Lmodel_2/model_1/map_features/model_1/stereo_embedding/BiasAdd/ReadVariableOp�Kmodel_2/model_1/map_features/model_1/stereo_embedding/MatMul/ReadVariableOp�Kmodel_2/model_1/sequential_29/sequential_27/dense_21/BiasAdd/ReadVariableOp�Jmodel_2/model_1/sequential_29/sequential_27/dense_21/MatMul/ReadVariableOp�Kmodel_2/model_1/sequential_29/sequential_28/dense_22/BiasAdd/ReadVariableOp�Jmodel_2/model_1/sequential_29/sequential_28/dense_22/MatMul/ReadVariableOp�?model_2/model_1/structured_readout/assert_equal_1/Assert/Assert�<model_2/model_1/structured_readout/assert_less/Assert/Assert�bmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert�ymodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert�Hmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Assert�dmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert�{model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert�Jmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Assert�dmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert�{model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert�Jmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Assert�dmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert�{model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert�Jmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Assert�dmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert�{model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert�Jmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Assert�dmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert�{model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert�Jmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Assert�dmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert�{model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert�Jmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Assert�3model_2/parse_example_2/assert_shapes/Assert/Assert�5model_2/parse_example_2/assert_shapes/Assert_1/Assert�5model_2/parse_example_2/assert_shapes/Assert_2/Assert�5model_2/parse_example_2/assert_shapes/Assert_3/Assertg
model_2/parse_example_2/zerosConst*
_output_shapes
:*
dtype0	*
valueB	R i
model_2/parse_example_2/zeros_1Const*
_output_shapes
:*
dtype0	*
valueB	R i
model_2/parse_example_2/zeros_2Const*
_output_shapes
:*
dtype0	*
valueB	R i
model_2/parse_example_2/zeros_3Const*
_output_shapes
:*
dtype0	*
valueB	R i
model_2/parse_example_2/zeros_4Const*
_output_shapes
:*
dtype0	*
valueB	R m
*model_2/parse_example_2/ParseExample/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 o
,model_2/parse_example_2/ParseExample/Const_1Const*
_output_shapes
: *
dtype0*
valueB |
9model_2/parse_example_2/ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB �
?model_2/parse_example_2/ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB �
>model_2/parse_example_2/ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*�
value�B�Bcontext/_mol_idBcontext/smilesBedges/_readout/shape.#sizeBedges/bond.#sizeB edges/interatomic_distance.#sizeBnodes/_readout.#sizeBnodes/atom.#size�
?model_2/parse_example_2/ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
:*
dtype0*�
value�B�Bedges/_readout/shape.#sourceBedges/_readout/shape.#targetBedges/bond.#sourceBedges/bond.#targetBedges/bond.bond_typeBedges/bond.distanceBedges/bond.is_conjugatedBedges/bond.normalized_distanceBedges/bond.rbf_distanceBedges/bond.stereoB"edges/interatomic_distance.#sourceB"edges/interatomic_distance.#targetB#edges/interatomic_distance.distanceB'edges/interatomic_distance.rbf_distanceBnodes/_readout.shapeBnodes/atom._atom_idxBnodes/atom.atom_symBnodes/atom.chiral_tagBnodes/atom.degreeBnodes/atom.formal_chargeBnodes/atom.hybridizationBnodes/atom.is_aromaticBnodes/atom.no_implicitBnodes/atom.num_HsB nodes/atom.num_radical_electronsBnodes/atom.shiftBnodes/atom.valence�
3model_2/parse_example_2/ParseExample/ParseExampleV2ParseExampleV2examplesBmodel_2/parse_example_2/ParseExample/ParseExampleV2/names:output:0Hmodel_2/parse_example_2/ParseExample/ParseExampleV2/sparse_keys:output:0Gmodel_2/parse_example_2/ParseExample/ParseExampleV2/dense_keys:output:0Hmodel_2/parse_example_2/ParseExample/ParseExampleV2/ragged_keys:output:03model_2/parse_example_2/ParseExample/Const:output:05model_2/parse_example_2/ParseExample/Const_1:output:0(model_2/parse_example_2/zeros_3:output:0(model_2/parse_example_2/zeros_2:output:0(model_2/parse_example_2/zeros_4:output:0&model_2/parse_example_2/zeros:output:0(model_2/parse_example_2/zeros_1:output:0*
Tdense
	2						*�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������*<
dense_shapes,
*:::::::*

num_sparse *5
ragged_split_types
2*5
ragged_value_types
2															*
sparse_types
 �
model_2/parse_example_2/ShapeShapeCmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:4*
T0*
_output_shapes
::��u
+model_2/parse_example_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-model_2/parse_example_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-model_2/parse_example_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%model_2/parse_example_2/strided_sliceStridedSlice&model_2/parse_example_2/Shape:output:04model_2/parse_example_2/strided_slice/stack:output:06model_2/parse_example_2/strided_slice/stack_1:output:06model_2/parse_example_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskx
'model_2/parse_example_2/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"����   e
#model_2/parse_example_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
model_2/parse_example_2/concatConcatV20model_2/parse_example_2/concat/values_0:output:0.model_2/parse_example_2/strided_slice:output:0,model_2/parse_example_2/concat/axis:output:0*
N*
T0*
_output_shapes
:�
model_2/parse_example_2/ReshapeReshapeCmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:4'model_2/parse_example_2/concat:output:0*
T0*'
_output_shapes
:���������d
"model_2/parse_example_2/floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :�
 model_2/parse_example_2/floordivFloorDivGmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:4+model_2/parse_example_2/floordiv/y:output:0*
T0*#
_output_shapes
:����������
Vmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Wmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/ShapeShape$model_2/parse_example_2/floordiv:z:0*
T0*
_output_shapes
::���
�model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
qmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Ymodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
[model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
[model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Smodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_sliceStridedSlice$model_2/parse_example_2/floordiv:z:0bmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack:output:0dmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_1:output:0dmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Kmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
Zmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/EqualEqual\model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:output:0Tmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:output:0*
T0*
_output_shapes
: �
Ymodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
`model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
`model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
Zmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/rangeRangeimodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0bmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0imodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: �
Xmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/AllAll^model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0cmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: �
amodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
cmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
cmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*m
valuedBb B\x (model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = �
cmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*e
value\BZ BTy (model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = �
imodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
imodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
imodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*m
valuedBb B\x (model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = �
imodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*e
value\BZ BTy (model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = �
bmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertAssertamodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/All:output:0rmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0:output:0rmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1:output:0rmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2:output:0\model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:output:0rmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4:output:0Tmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:output:0*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
[model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
]model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
]model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1StridedSlice$model_2/parse_example_2/floordiv:z:0dmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack:output:0fmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0fmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*
end_mask�
[model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
]model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
]model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2StridedSlice$model_2/parse_example_2/floordiv:z:0dmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack:output:0fmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0fmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask�
Imodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/subSub^model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1:output:0^model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0*#
_output_shapes
:����������
_model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
umodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualhmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/Const:output:0Mmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:z:0*
T0*#
_output_shapes
:����������
qmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
omodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAllymodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0zmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: �
xmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
zmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
zmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*c
valueZBX BRx (model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = �
�model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
�model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
�model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*c
valueZBX BRx (model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = �
ymodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertAssertxmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0�model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0:output:0�model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1:output:0�model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2:output:0Mmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:z:0c^model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert*
T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Xmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/control_dependencyIdentity$model_2/parse_example_2/floordiv:z:0c^model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assertz^model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assertr^model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0*3
_class)
'%loc:@model_2/parse_example_2/floordiv*#
_output_shapes
:����������
1model_2/parse_example_2/RaggedFromRowSplits/ShapeShape(model_2/parse_example_2/Reshape:output:0*
T0*
_output_shapes
::���
?model_2/parse_example_2/RaggedFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Amodel_2/parse_example_2/RaggedFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Amodel_2/parse_example_2/RaggedFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9model_2/parse_example_2/RaggedFromRowSplits/strided_sliceStridedSlice:model_2/parse_example_2/RaggedFromRowSplits/Shape:output:0Hmodel_2/parse_example_2/RaggedFromRowSplits/strided_slice/stack:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits/strided_slice/stack_1:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Amodel_2/parse_example_2/RaggedFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Cmodel_2/parse_example_2/RaggedFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Cmodel_2/parse_example_2/RaggedFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
;model_2/parse_example_2/RaggedFromRowSplits/strided_slice_1StridedSliceamodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/control_dependency:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits/strided_slice_1/stack:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits/strided_slice_1/stack_1:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
@model_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/EqualEqualDmodel_2/parse_example_2/RaggedFromRowSplits/strided_slice_1:output:0Bmodel_2/parse_example_2/RaggedFromRowSplits/strided_slice:output:0*
T0*
_output_shapes
: �
?model_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
Fmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
Fmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
@model_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/rangeRangeOmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/range/start:output:0Hmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Rank:output:0Omodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: �
>model_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/AllAllDmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Equal:z:0Imodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: �
Gmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Imodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Imodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*U
valueLBJ BDx (model_2/parse_example_2/RaggedFromRowSplits/strided_slice_1:0) = �
Imodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*S
valueJBH BBy (model_2/parse_example_2/RaggedFromRowSplits/strided_slice:0) = �
Omodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Omodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Omodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*U
valueLBJ BDx (model_2/parse_example_2/RaggedFromRowSplits/strided_slice_1:0) = �
Omodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*S
valueJBH BBy (model_2/parse_example_2/RaggedFromRowSplits/strided_slice:0) = �
Hmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/AssertAssertGmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/All:output:0Xmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Assert/data_0:output:0Xmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Assert/data_1:output:0Xmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Assert/data_2:output:0Dmodel_2/parse_example_2/RaggedFromRowSplits/strided_slice_1:output:0Xmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Assert/data_4:output:0Bmodel_2/parse_example_2/RaggedFromRowSplits/strided_slice:output:0z^model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Emodel_2/parse_example_2/RaggedFromRowSplits/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Fmodel_2/parse_example_2/RaggedFromRowSplits/assert_rank_at_least/ShapeShape(model_2/parse_example_2/Reshape:output:0*
T0*
_output_shapes
::���
omodel_2/parse_example_2/RaggedFromRowSplits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
`model_2/parse_example_2/RaggedFromRowSplits/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
>model_2/parse_example_2/RaggedFromRowSplits/control_dependencyIdentityamodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/control_dependency:output:0I^model_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Asserta^model_2/parse_example_2/RaggedFromRowSplits/assert_rank_at_least/static_checks_determined_all_ok*
T0*3
_class)
'%loc:@model_2/parse_example_2/floordiv*#
_output_shapes
:����������
model_2/parse_example_2/Shape_1ShapeCmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:9*
T0*
_output_shapes
::��w
-model_2/parse_example_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/model_2/parse_example_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/model_2/parse_example_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'model_2/parse_example_2/strided_slice_1StridedSlice(model_2/parse_example_2/Shape_1:output:06model_2/parse_example_2/strided_slice_1/stack:output:08model_2/parse_example_2/strided_slice_1/stack_1:output:08model_2/parse_example_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskz
)model_2/parse_example_2/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB"����   g
%model_2/parse_example_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model_2/parse_example_2/concat_1ConcatV22model_2/parse_example_2/concat_1/values_0:output:00model_2/parse_example_2/strided_slice_1:output:0.model_2/parse_example_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
!model_2/parse_example_2/Reshape_1ReshapeCmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:9)model_2/parse_example_2/concat_1:output:0*
T0*'
_output_shapes
:���������f
$model_2/parse_example_2/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
"model_2/parse_example_2/floordiv_1FloorDivGmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:9-model_2/parse_example_2/floordiv_1/y:output:0*
T0*#
_output_shapes
:����������
Xmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Ymodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/ShapeShape&model_2/parse_example_2/floordiv_1:z:0*
T0*
_output_shapes
::���
�model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
smodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
[model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
]model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
]model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_sliceStridedSlice&model_2/parse_example_2/floordiv_1:z:0dmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack:output:0fmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_1:output:0fmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Mmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
\model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/EqualEqual^model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:output:0Vmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:output:0*
T0*
_output_shapes
: �
[model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
bmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
bmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
\model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/rangeRangekmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0dmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0kmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: �
Zmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/AllAll`model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0emodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: �
cmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
emodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
emodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*o
valuefBd B^x (model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:0) = �
emodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*g
value^B\ BVy (model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:0) = �
kmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
kmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
kmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*o
valuefBd B^x (model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:0) = �
kmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*g
value^B\ BVy (model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:0) = �
dmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertAssertcmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/All:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2:output:0^model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4:output:0Vmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:output:0I^model_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
]model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
_model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
_model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1StridedSlice&model_2/parse_example_2/floordiv_1:z:0fmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*
end_mask�
]model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
_model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
_model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2StridedSlice&model_2/parse_example_2/floordiv_1:z:0fmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask�
Kmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/subSub`model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1:output:0`model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0*#
_output_shapes
:����������
amodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
wmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualjmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/Const:output:0Omodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:z:0*
T0*#
_output_shapes
:����������
smodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
qmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAll{model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0|model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: �
zmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
|model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
|model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*e
value\BZ BTx (model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:0) = �
�model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
�model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
�model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*e
value\BZ BTx (model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:0) = �
{model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertAssertzmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0�model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0:output:0�model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1:output:0�model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2:output:0Omodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:z:0e^model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert*
T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Zmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/control_dependencyIdentity&model_2/parse_example_2/floordiv_1:z:0e^model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert|^model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assertt^model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0*5
_class+
)'loc:@model_2/parse_example_2/floordiv_1*#
_output_shapes
:����������
3model_2/parse_example_2/RaggedFromRowSplits_1/ShapeShape*model_2/parse_example_2/Reshape_1:output:0*
T0*
_output_shapes
::���
Amodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Cmodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
;model_2/parse_example_2/RaggedFromRowSplits_1/strided_sliceStridedSlice<model_2/parse_example_2/RaggedFromRowSplits_1/Shape:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice/stack:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice/stack_1:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Cmodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Emodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Emodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=model_2/parse_example_2/RaggedFromRowSplits_1/strided_slice_1StridedSlicecmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/control_dependency:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice_1/stack:output:0Nmodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice_1/stack_1:output:0Nmodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/EqualEqualFmodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice_1:output:0Dmodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice:output:0*
T0*
_output_shapes
: �
Amodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
Hmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
Hmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
Bmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/rangeRangeQmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/range/start:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Rank:output:0Qmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/range/delta:output:0*
_output_shapes
: �
@model_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/AllAllFmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Equal:z:0Kmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/range:output:0*
_output_shapes
: �
Imodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Kmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Kmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (model_2/parse_example_2/RaggedFromRowSplits_1/strided_slice_1:0) = �
Kmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (model_2/parse_example_2/RaggedFromRowSplits_1/strided_slice:0) = �
Qmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Qmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Qmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (model_2/parse_example_2/RaggedFromRowSplits_1/strided_slice_1:0) = �
Qmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (model_2/parse_example_2/RaggedFromRowSplits_1/strided_slice:0) = �
Jmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/AssertAssertImodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/All:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Assert/data_0:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Assert/data_1:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Assert/data_2:output:0Fmodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice_1:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Assert/data_4:output:0Dmodel_2/parse_example_2/RaggedFromRowSplits_1/strided_slice:output:0|^model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Gmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Hmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_rank_at_least/ShapeShape*model_2/parse_example_2/Reshape_1:output:0*
T0*
_output_shapes
::���
qmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
bmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
@model_2/parse_example_2/RaggedFromRowSplits_1/control_dependencyIdentitycmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/control_dependency:output:0K^model_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Assertc^model_2/parse_example_2/RaggedFromRowSplits_1/assert_rank_at_least/static_checks_determined_all_ok*
T0*5
_class+
)'loc:@model_2/parse_example_2/floordiv_1*#
_output_shapes
:����������
model_2/parse_example_2/Shape_2ShapeDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:14*
T0*
_output_shapes
::��w
-model_2/parse_example_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/model_2/parse_example_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/model_2/parse_example_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'model_2/parse_example_2/strided_slice_2StridedSlice(model_2/parse_example_2/Shape_2:output:06model_2/parse_example_2/strided_slice_2/stack:output:08model_2/parse_example_2/strided_slice_2/stack_1:output:08model_2/parse_example_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskz
)model_2/parse_example_2/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB"����   g
%model_2/parse_example_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model_2/parse_example_2/concat_2ConcatV22model_2/parse_example_2/concat_2/values_0:output:00model_2/parse_example_2/strided_slice_2:output:0.model_2/parse_example_2/concat_2/axis:output:0*
N*
T0*
_output_shapes
:�
!model_2/parse_example_2/Reshape_2ReshapeDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:14)model_2/parse_example_2/concat_2:output:0*
T0*'
_output_shapes
:���������f
$model_2/parse_example_2/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
"model_2/parse_example_2/floordiv_2FloorDivHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:14-model_2/parse_example_2/floordiv_2/y:output:0*
T0*#
_output_shapes
:����������
Xmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Ymodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/ShapeShape&model_2/parse_example_2/floordiv_2:z:0*
T0*
_output_shapes
::���
�model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
smodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
[model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
]model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
]model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_sliceStridedSlice&model_2/parse_example_2/floordiv_2:z:0dmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack:output:0fmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_1:output:0fmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Mmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
\model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/EqualEqual^model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:output:0Vmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:output:0*
T0*
_output_shapes
: �
[model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
bmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
bmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
\model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/rangeRangekmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0dmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0kmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: �
Zmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/AllAll`model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0emodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: �
cmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
emodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
emodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*o
valuefBd B^x (model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:0) = �
emodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*g
value^B\ BVy (model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:0) = �
kmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
kmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
kmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*o
valuefBd B^x (model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:0) = �
kmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*g
value^B\ BVy (model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:0) = �
dmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertAssertcmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/All:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2:output:0^model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4:output:0Vmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:output:0K^model_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
]model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
_model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
_model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1StridedSlice&model_2/parse_example_2/floordiv_2:z:0fmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*
end_mask�
]model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
_model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
_model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2StridedSlice&model_2/parse_example_2/floordiv_2:z:0fmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask�
Kmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/subSub`model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1:output:0`model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0*#
_output_shapes
:����������
amodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
wmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualjmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/Const:output:0Omodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:z:0*
T0*#
_output_shapes
:����������
smodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
qmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAll{model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0|model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: �
zmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
|model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
|model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*e
value\BZ BTx (model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:0) = �
�model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
�model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
�model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*e
value\BZ BTx (model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:0) = �
{model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertAssertzmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0�model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0:output:0�model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1:output:0�model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2:output:0Omodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:z:0e^model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert*
T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Zmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/control_dependencyIdentity&model_2/parse_example_2/floordiv_2:z:0e^model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert|^model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assertt^model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0*5
_class+
)'loc:@model_2/parse_example_2/floordiv_2*#
_output_shapes
:����������
3model_2/parse_example_2/RaggedFromRowSplits_2/ShapeShape*model_2/parse_example_2/Reshape_2:output:0*
T0*
_output_shapes
::���
Amodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Cmodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
;model_2/parse_example_2/RaggedFromRowSplits_2/strided_sliceStridedSlice<model_2/parse_example_2/RaggedFromRowSplits_2/Shape:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice/stack:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice/stack_1:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Cmodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Emodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Emodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=model_2/parse_example_2/RaggedFromRowSplits_2/strided_slice_1StridedSlicecmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/control_dependency:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice_1/stack:output:0Nmodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice_1/stack_1:output:0Nmodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/EqualEqualFmodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice_1:output:0Dmodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice:output:0*
T0*
_output_shapes
: �
Amodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
Hmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
Hmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
Bmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/rangeRangeQmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/range/start:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Rank:output:0Qmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/range/delta:output:0*
_output_shapes
: �
@model_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/AllAllFmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Equal:z:0Kmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/range:output:0*
_output_shapes
: �
Imodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Kmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Kmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (model_2/parse_example_2/RaggedFromRowSplits_2/strided_slice_1:0) = �
Kmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (model_2/parse_example_2/RaggedFromRowSplits_2/strided_slice:0) = �
Qmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Qmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Qmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (model_2/parse_example_2/RaggedFromRowSplits_2/strided_slice_1:0) = �
Qmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (model_2/parse_example_2/RaggedFromRowSplits_2/strided_slice:0) = �
Jmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/AssertAssertImodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/All:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Assert/data_0:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Assert/data_1:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Assert/data_2:output:0Fmodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice_1:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Assert/data_4:output:0Dmodel_2/parse_example_2/RaggedFromRowSplits_2/strided_slice:output:0|^model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Gmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Hmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_rank_at_least/ShapeShape*model_2/parse_example_2/Reshape_2:output:0*
T0*
_output_shapes
::���
qmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
bmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
@model_2/parse_example_2/RaggedFromRowSplits_2/control_dependencyIdentitycmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/control_dependency:output:0K^model_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Assertc^model_2/parse_example_2/RaggedFromRowSplits_2/assert_rank_at_least/static_checks_determined_all_ok*
T0*5
_class+
)'loc:@model_2/parse_example_2/floordiv_2*#
_output_shapes
:����������
model_2/parse_example_2/Shape_3Shape*model_2/parse_example_2/Reshape_2:output:0*
T0*
_output_shapes
::��w
-model_2/parse_example_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/model_2/parse_example_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/model_2/parse_example_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'model_2/parse_example_2/strided_slice_3StridedSlice(model_2/parse_example_2/Shape_3:output:06model_2/parse_example_2/strided_slice_3/stack:output:08model_2/parse_example_2/strided_slice_3/stack_1:output:08model_2/parse_example_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskz
)model_2/parse_example_2/concat_3/values_0Const*
_output_shapes
:*
dtype0*
valueB"����   g
%model_2/parse_example_2/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model_2/parse_example_2/concat_3ConcatV22model_2/parse_example_2/concat_3/values_0:output:00model_2/parse_example_2/strided_slice_3:output:0.model_2/parse_example_2/concat_3/axis:output:0*
N*
T0*
_output_shapes
:�
!model_2/parse_example_2/Reshape_3Reshape*model_2/parse_example_2/Reshape_2:output:0)model_2/parse_example_2/concat_3:output:0*
T0*+
_output_shapes
:���������f
$model_2/parse_example_2/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :�
"model_2/parse_example_2/floordiv_3FloorDivImodel_2/parse_example_2/RaggedFromRowSplits_2/control_dependency:output:0-model_2/parse_example_2/floordiv_3/y:output:0*
T0*#
_output_shapes
:����������
Xmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Ymodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/ShapeShape&model_2/parse_example_2/floordiv_3:z:0*
T0*
_output_shapes
::���
�model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
smodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
[model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
]model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
]model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_sliceStridedSlice&model_2/parse_example_2/floordiv_3:z:0dmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack:output:0fmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_1:output:0fmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Mmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
\model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/EqualEqual^model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:output:0Vmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:output:0*
T0*
_output_shapes
: �
[model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
bmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
bmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
\model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/rangeRangekmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0dmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0kmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: �
Zmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/AllAll`model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0emodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: �
cmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
emodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
emodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*o
valuefBd B^x (model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:0) = �
emodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*g
value^B\ BVy (model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:0) = �
kmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
kmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
kmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*o
valuefBd B^x (model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:0) = �
kmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*g
value^B\ BVy (model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:0) = �
dmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertAssertcmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/All:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2:output:0^model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4:output:0Vmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:output:0K^model_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
]model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
_model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
_model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1StridedSlice&model_2/parse_example_2/floordiv_3:z:0fmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*
end_mask�
]model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
_model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
_model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2StridedSlice&model_2/parse_example_2/floordiv_3:z:0fmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask�
Kmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/subSub`model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1:output:0`model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0*#
_output_shapes
:����������
amodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
wmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualjmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/Const:output:0Omodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:z:0*
T0*#
_output_shapes
:����������
smodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
qmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAll{model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0|model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: �
zmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
|model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
|model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*e
value\BZ BTx (model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:0) = �
�model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
�model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
�model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*e
value\BZ BTx (model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:0) = �
{model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertAssertzmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0�model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0:output:0�model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1:output:0�model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2:output:0Omodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:z:0e^model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert*
T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Zmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/control_dependencyIdentity&model_2/parse_example_2/floordiv_3:z:0e^model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert|^model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assertt^model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0*5
_class+
)'loc:@model_2/parse_example_2/floordiv_3*#
_output_shapes
:����������
3model_2/parse_example_2/RaggedFromRowSplits_3/ShapeShape*model_2/parse_example_2/Reshape_3:output:0*
T0*
_output_shapes
::���
Amodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Cmodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
;model_2/parse_example_2/RaggedFromRowSplits_3/strided_sliceStridedSlice<model_2/parse_example_2/RaggedFromRowSplits_3/Shape:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice/stack:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice/stack_1:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Cmodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Emodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Emodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=model_2/parse_example_2/RaggedFromRowSplits_3/strided_slice_1StridedSlicecmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/control_dependency:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice_1/stack:output:0Nmodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice_1/stack_1:output:0Nmodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/EqualEqualFmodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice_1:output:0Dmodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice:output:0*
T0*
_output_shapes
: �
Amodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
Hmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
Hmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
Bmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/rangeRangeQmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/range/start:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Rank:output:0Qmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/range/delta:output:0*
_output_shapes
: �
@model_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/AllAllFmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Equal:z:0Kmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/range:output:0*
_output_shapes
: �
Imodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Kmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Kmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (model_2/parse_example_2/RaggedFromRowSplits_3/strided_slice_1:0) = �
Kmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (model_2/parse_example_2/RaggedFromRowSplits_3/strided_slice:0) = �
Qmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Qmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Qmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (model_2/parse_example_2/RaggedFromRowSplits_3/strided_slice_1:0) = �
Qmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (model_2/parse_example_2/RaggedFromRowSplits_3/strided_slice:0) = �
Jmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/AssertAssertImodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/All:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Assert/data_0:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Assert/data_1:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Assert/data_2:output:0Fmodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice_1:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Assert/data_4:output:0Dmodel_2/parse_example_2/RaggedFromRowSplits_3/strided_slice:output:0|^model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Gmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Hmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_rank_at_least/ShapeShape*model_2/parse_example_2/Reshape_3:output:0*
T0*
_output_shapes
::���
qmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
bmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
@model_2/parse_example_2/RaggedFromRowSplits_3/control_dependencyIdentitycmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/control_dependency:output:0K^model_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Assertc^model_2/parse_example_2/RaggedFromRowSplits_3/assert_rank_at_least/static_checks_determined_all_ok*
T0*5
_class+
)'loc:@model_2/parse_example_2/floordiv_3*#
_output_shapes
:����������
model_2/parse_example_2/Shape_4ShapeDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:16*
T0*
_output_shapes
::��w
-model_2/parse_example_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/model_2/parse_example_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/model_2/parse_example_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'model_2/parse_example_2/strided_slice_4StridedSlice(model_2/parse_example_2/Shape_4:output:06model_2/parse_example_2/strided_slice_4/stack:output:08model_2/parse_example_2/strided_slice_4/stack_1:output:08model_2/parse_example_2/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskz
)model_2/parse_example_2/concat_4/values_0Const*
_output_shapes
:*
dtype0*
valueB"����   g
%model_2/parse_example_2/concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model_2/parse_example_2/concat_4ConcatV22model_2/parse_example_2/concat_4/values_0:output:00model_2/parse_example_2/strided_slice_4:output:0.model_2/parse_example_2/concat_4/axis:output:0*
N*
T0*
_output_shapes
:�
!model_2/parse_example_2/Reshape_4ReshapeDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:16)model_2/parse_example_2/concat_4:output:0*
T0*'
_output_shapes
:���������f
$model_2/parse_example_2/floordiv_4/yConst*
_output_shapes
: *
dtype0*
value	B :�
"model_2/parse_example_2/floordiv_4FloorDivHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:16-model_2/parse_example_2/floordiv_4/y:output:0*
T0*#
_output_shapes
:����������
Xmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Ymodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_rank/ShapeShape&model_2/parse_example_2/floordiv_4:z:0*
T0*
_output_shapes
::���
�model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
smodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
[model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
]model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
]model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_sliceStridedSlice&model_2/parse_example_2/floordiv_4:z:0dmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice/stack:output:0fmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice/stack_1:output:0fmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Mmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
\model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/EqualEqual^model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice:output:0Vmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/Const:output:0*
T0*
_output_shapes
: �
[model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
bmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
bmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
\model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/rangeRangekmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0dmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0kmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: �
Zmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/AllAll`model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0emodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: �
cmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
emodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
emodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*o
valuefBd B^x (model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice:0) = �
emodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*g
value^B\ BVy (model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/Const:0) = �
kmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
kmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
kmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*o
valuefBd B^x (model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice:0) = �
kmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*g
value^B\ BVy (model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/Const:0) = �
dmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertAssertcmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/All:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2:output:0^model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4:output:0Vmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/Const:output:0K^model_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
]model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
_model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
_model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_1StridedSlice&model_2/parse_example_2/floordiv_4:z:0fmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_1/stack:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*
end_mask�
]model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
_model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
_model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_2StridedSlice&model_2/parse_example_2/floordiv_4:z:0fmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_2/stack:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask�
Kmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/subSub`model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_1:output:0`model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0*#
_output_shapes
:����������
amodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
wmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualjmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/Const:output:0Omodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/sub:z:0*
T0*#
_output_shapes
:����������
smodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
qmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAll{model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0|model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: �
zmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
|model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
|model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*e
value\BZ BTx (model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/sub:0) = �
�model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
�model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
�model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*e
value\BZ BTx (model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/sub:0) = �
{model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertAssertzmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0�model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0:output:0�model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1:output:0�model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2:output:0Omodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/sub:z:0e^model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert*
T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Zmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/control_dependencyIdentity&model_2/parse_example_2/floordiv_4:z:0e^model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert|^model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assertt^model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0*5
_class+
)'loc:@model_2/parse_example_2/floordiv_4*#
_output_shapes
:����������
3model_2/parse_example_2/RaggedFromRowSplits_4/ShapeShape*model_2/parse_example_2/Reshape_4:output:0*
T0*
_output_shapes
::���
Amodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Cmodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
;model_2/parse_example_2/RaggedFromRowSplits_4/strided_sliceStridedSlice<model_2/parse_example_2/RaggedFromRowSplits_4/Shape:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice/stack:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice/stack_1:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Cmodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Emodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Emodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=model_2/parse_example_2/RaggedFromRowSplits_4/strided_slice_1StridedSlicecmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/control_dependency:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice_1/stack:output:0Nmodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice_1/stack_1:output:0Nmodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/EqualEqualFmodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice_1:output:0Dmodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice:output:0*
T0*
_output_shapes
: �
Amodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
Hmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
Hmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
Bmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/rangeRangeQmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/range/start:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Rank:output:0Qmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/range/delta:output:0*
_output_shapes
: �
@model_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/AllAllFmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Equal:z:0Kmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/range:output:0*
_output_shapes
: �
Imodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Kmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Kmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (model_2/parse_example_2/RaggedFromRowSplits_4/strided_slice_1:0) = �
Kmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (model_2/parse_example_2/RaggedFromRowSplits_4/strided_slice:0) = �
Qmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Qmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Qmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (model_2/parse_example_2/RaggedFromRowSplits_4/strided_slice_1:0) = �
Qmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (model_2/parse_example_2/RaggedFromRowSplits_4/strided_slice:0) = �
Jmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/AssertAssertImodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/All:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Assert/data_0:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Assert/data_1:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Assert/data_2:output:0Fmodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice_1:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Assert/data_4:output:0Dmodel_2/parse_example_2/RaggedFromRowSplits_4/strided_slice:output:0|^model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Gmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Hmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_rank_at_least/ShapeShape*model_2/parse_example_2/Reshape_4:output:0*
T0*
_output_shapes
::���
qmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
bmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
@model_2/parse_example_2/RaggedFromRowSplits_4/control_dependencyIdentitycmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/control_dependency:output:0K^model_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Assertc^model_2/parse_example_2/RaggedFromRowSplits_4/assert_rank_at_least/static_checks_determined_all_ok*
T0*5
_class+
)'loc:@model_2/parse_example_2/floordiv_4*#
_output_shapes
:����������
model_2/parse_example_2/Shape_5ShapeDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:17*
T0*
_output_shapes
::��w
-model_2/parse_example_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/model_2/parse_example_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/model_2/parse_example_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'model_2/parse_example_2/strided_slice_5StridedSlice(model_2/parse_example_2/Shape_5:output:06model_2/parse_example_2/strided_slice_5/stack:output:08model_2/parse_example_2/strided_slice_5/stack_1:output:08model_2/parse_example_2/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskz
)model_2/parse_example_2/concat_5/values_0Const*
_output_shapes
:*
dtype0*
valueB"����	   g
%model_2/parse_example_2/concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model_2/parse_example_2/concat_5ConcatV22model_2/parse_example_2/concat_5/values_0:output:00model_2/parse_example_2/strided_slice_5:output:0.model_2/parse_example_2/concat_5/axis:output:0*
N*
T0*
_output_shapes
:�
!model_2/parse_example_2/Reshape_5ReshapeDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:17)model_2/parse_example_2/concat_5:output:0*
T0*'
_output_shapes
:���������	f
$model_2/parse_example_2/floordiv_5/yConst*
_output_shapes
: *
dtype0*
value	B :	�
"model_2/parse_example_2/floordiv_5FloorDivHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:17-model_2/parse_example_2/floordiv_5/y:output:0*
T0*#
_output_shapes
:����������
Xmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Ymodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_rank/ShapeShape&model_2/parse_example_2/floordiv_5:z:0*
T0*
_output_shapes
::���
�model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
smodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
[model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
]model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
]model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_sliceStridedSlice&model_2/parse_example_2/floordiv_5:z:0dmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice/stack:output:0fmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice/stack_1:output:0fmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Mmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
\model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/EqualEqual^model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice:output:0Vmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/Const:output:0*
T0*
_output_shapes
: �
[model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
bmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
bmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
\model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/rangeRangekmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0dmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0kmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: �
Zmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/AllAll`model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0emodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: �
cmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
emodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
emodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*o
valuefBd B^x (model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice:0) = �
emodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*g
value^B\ BVy (model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/Const:0) = �
kmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
kmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
kmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*o
valuefBd B^x (model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice:0) = �
kmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*g
value^B\ BVy (model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/Const:0) = �
dmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertAssertcmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/All:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2:output:0^model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4:output:0Vmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/Const:output:0K^model_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
]model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
_model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
_model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_1StridedSlice&model_2/parse_example_2/floordiv_5:z:0fmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_1/stack:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*
end_mask�
]model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
_model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
_model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_2StridedSlice&model_2/parse_example_2/floordiv_5:z:0fmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_2/stack:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask�
Kmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/subSub`model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_1:output:0`model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0*#
_output_shapes
:����������
amodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
wmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualjmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/Const:output:0Omodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/sub:z:0*
T0*#
_output_shapes
:����������
smodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
qmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAll{model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0|model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: �
zmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
|model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
|model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*e
value\BZ BTx (model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/sub:0) = �
�model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
�model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
�model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*e
value\BZ BTx (model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/sub:0) = �
{model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertAssertzmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0�model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0:output:0�model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1:output:0�model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2:output:0Omodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/sub:z:0e^model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert*
T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Zmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/control_dependencyIdentity&model_2/parse_example_2/floordiv_5:z:0e^model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert|^model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assertt^model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0*5
_class+
)'loc:@model_2/parse_example_2/floordiv_5*#
_output_shapes
:����������
3model_2/parse_example_2/RaggedFromRowSplits_5/ShapeShape*model_2/parse_example_2/Reshape_5:output:0*
T0*
_output_shapes
::���
Amodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Cmodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
;model_2/parse_example_2/RaggedFromRowSplits_5/strided_sliceStridedSlice<model_2/parse_example_2/RaggedFromRowSplits_5/Shape:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice/stack:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice/stack_1:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Cmodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Emodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Emodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=model_2/parse_example_2/RaggedFromRowSplits_5/strided_slice_1StridedSlicecmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/control_dependency:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice_1/stack:output:0Nmodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice_1/stack_1:output:0Nmodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/EqualEqualFmodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice_1:output:0Dmodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice:output:0*
T0*
_output_shapes
: �
Amodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
Hmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
Hmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
Bmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/rangeRangeQmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/range/start:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Rank:output:0Qmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/range/delta:output:0*
_output_shapes
: �
@model_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/AllAllFmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Equal:z:0Kmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/range:output:0*
_output_shapes
: �
Imodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Kmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Kmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (model_2/parse_example_2/RaggedFromRowSplits_5/strided_slice_1:0) = �
Kmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (model_2/parse_example_2/RaggedFromRowSplits_5/strided_slice:0) = �
Qmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Qmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Qmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (model_2/parse_example_2/RaggedFromRowSplits_5/strided_slice_1:0) = �
Qmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (model_2/parse_example_2/RaggedFromRowSplits_5/strided_slice:0) = �
Jmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/AssertAssertImodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/All:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Assert/data_0:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Assert/data_1:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Assert/data_2:output:0Fmodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice_1:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Assert/data_4:output:0Dmodel_2/parse_example_2/RaggedFromRowSplits_5/strided_slice:output:0|^model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Gmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Hmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_rank_at_least/ShapeShape*model_2/parse_example_2/Reshape_5:output:0*
T0*
_output_shapes
::���
qmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
bmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
@model_2/parse_example_2/RaggedFromRowSplits_5/control_dependencyIdentitycmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/control_dependency:output:0K^model_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Assertc^model_2/parse_example_2/RaggedFromRowSplits_5/assert_rank_at_least/static_checks_determined_all_ok*
T0*5
_class+
)'loc:@model_2/parse_example_2/floordiv_5*#
_output_shapes
:����������
model_2/parse_example_2/Shape_6ShapeDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:20*
T0*
_output_shapes
::��w
-model_2/parse_example_2/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/model_2/parse_example_2/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/model_2/parse_example_2/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'model_2/parse_example_2/strided_slice_6StridedSlice(model_2/parse_example_2/Shape_6:output:06model_2/parse_example_2/strided_slice_6/stack:output:08model_2/parse_example_2/strided_slice_6/stack_1:output:08model_2/parse_example_2/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskz
)model_2/parse_example_2/concat_6/values_0Const*
_output_shapes
:*
dtype0*
valueB"����	   g
%model_2/parse_example_2/concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model_2/parse_example_2/concat_6ConcatV22model_2/parse_example_2/concat_6/values_0:output:00model_2/parse_example_2/strided_slice_6:output:0.model_2/parse_example_2/concat_6/axis:output:0*
N*
T0*
_output_shapes
:�
!model_2/parse_example_2/Reshape_6ReshapeDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:20)model_2/parse_example_2/concat_6:output:0*
T0*'
_output_shapes
:���������	f
$model_2/parse_example_2/floordiv_6/yConst*
_output_shapes
: *
dtype0*
value	B :	�
"model_2/parse_example_2/floordiv_6FloorDivHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:20-model_2/parse_example_2/floordiv_6/y:output:0*
T0*#
_output_shapes
:����������
Xmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Ymodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_rank/ShapeShape&model_2/parse_example_2/floordiv_6:z:0*
T0*
_output_shapes
::���
�model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
smodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
[model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
]model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
]model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Umodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_sliceStridedSlice&model_2/parse_example_2/floordiv_6:z:0dmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice/stack:output:0fmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice/stack_1:output:0fmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Mmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
\model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/EqualEqual^model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice:output:0Vmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/Const:output:0*
T0*
_output_shapes
: �
[model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
bmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
bmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
\model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/rangeRangekmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0dmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0kmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: �
Zmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/AllAll`model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0emodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: �
cmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
emodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
emodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*o
valuefBd B^x (model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice:0) = �
emodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*g
value^B\ BVy (model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/Const:0) = �
kmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero�
kmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
kmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*o
valuefBd B^x (model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice:0) = �
kmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*g
value^B\ BVy (model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/Const:0) = �
dmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertAssertcmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/All:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_0:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_1:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_2:output:0^model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice:output:0tmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert/data_4:output:0Vmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/Const:output:0K^model_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
]model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
_model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
_model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_1StridedSlice&model_2/parse_example_2/floordiv_6:z:0fmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_1/stack:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*
end_mask�
]model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
_model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
_model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Wmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_2StridedSlice&model_2/parse_example_2/floordiv_6:z:0fmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_2/stack:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0hmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask�
Kmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/subSub`model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_1:output:0`model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0*#
_output_shapes
:����������
amodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
wmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualjmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/Const:output:0Omodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/sub:z:0*
T0*#
_output_shapes
:����������
smodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
qmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAll{model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0|model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: �
zmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
|model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
|model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*e
value\BZ BTx (model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/sub:0) = �
�model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  �
�model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:�
�model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*e
value\BZ BTx (model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/sub:0) = �
{model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertAssertzmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0�model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_0:output:0�model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_1:output:0�model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert/data_2:output:0Omodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/sub:z:0e^model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert*
T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Zmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/control_dependencyIdentity&model_2/parse_example_2/floordiv_6:z:0e^model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert|^model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assertt^model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0*5
_class+
)'loc:@model_2/parse_example_2/floordiv_6*#
_output_shapes
:����������
3model_2/parse_example_2/RaggedFromRowSplits_6/ShapeShape*model_2/parse_example_2/Reshape_6:output:0*
T0*
_output_shapes
::���
Amodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Cmodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
;model_2/parse_example_2/RaggedFromRowSplits_6/strided_sliceStridedSlice<model_2/parse_example_2/RaggedFromRowSplits_6/Shape:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice/stack:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice/stack_1:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Cmodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Emodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Emodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=model_2/parse_example_2/RaggedFromRowSplits_6/strided_slice_1StridedSlicecmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/control_dependency:output:0Lmodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice_1/stack:output:0Nmodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice_1/stack_1:output:0Nmodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Bmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/EqualEqualFmodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice_1:output:0Dmodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice:output:0*
T0*
_output_shapes
: �
Amodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
Hmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
Hmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
Bmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/rangeRangeQmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/range/start:output:0Jmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Rank:output:0Qmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/range/delta:output:0*
_output_shapes
: �
@model_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/AllAllFmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Equal:z:0Kmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/range:output:0*
_output_shapes
: �
Imodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Kmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Kmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (model_2/parse_example_2/RaggedFromRowSplits_6/strided_slice_1:0) = �
Kmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (model_2/parse_example_2/RaggedFromRowSplits_6/strided_slice:0) = �
Qmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor�
Qmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Qmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (model_2/parse_example_2/RaggedFromRowSplits_6/strided_slice_1:0) = �
Qmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (model_2/parse_example_2/RaggedFromRowSplits_6/strided_slice:0) = �
Jmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/AssertAssertImodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/All:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Assert/data_0:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Assert/data_1:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Assert/data_2:output:0Fmodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice_1:output:0Zmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Assert/data_4:output:0Dmodel_2/parse_example_2/RaggedFromRowSplits_6/strided_slice:output:0|^model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Gmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :�
Hmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_rank_at_least/ShapeShape*model_2/parse_example_2/Reshape_6:output:0*
T0*
_output_shapes
::���
qmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
bmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
@model_2/parse_example_2/RaggedFromRowSplits_6/control_dependencyIdentitycmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/control_dependency:output:0K^model_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Assertc^model_2/parse_example_2/RaggedFromRowSplits_6/assert_rank_at_least/static_checks_determined_all_ok*
T0*5
_class+
)'loc:@model_2/parse_example_2/floordiv_6*#
_output_shapes
:����������
+model_2/parse_example_2/assert_shapes/ShapeShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:2*
T0	*
_output_shapes
::���
9model_2/parse_example_2/assert_shapes/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
;model_2/parse_example_2/assert_shapes/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;model_2/parse_example_2/assert_shapes/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3model_2/parse_example_2/assert_shapes/strided_sliceStridedSlice4model_2/parse_example_2/assert_shapes/Shape:output:0Bmodel_2/parse_example_2/assert_shapes/strided_slice/stack:output:0Dmodel_2/parse_example_2/assert_shapes/strided_slice/stack_1:output:0Dmodel_2/parse_example_2/assert_shapes/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
-model_2/parse_example_2/assert_shapes/Shape_1ShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:3*
T0	*
_output_shapes
::���
;model_2/parse_example_2/assert_shapes/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
=model_2/parse_example_2/assert_shapes/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=model_2/parse_example_2/assert_shapes/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5model_2/parse_example_2/assert_shapes/strided_slice_1StridedSlice6model_2/parse_example_2/assert_shapes/Shape_1:output:0Dmodel_2/parse_example_2/assert_shapes/strided_slice_1/stack:output:0Fmodel_2/parse_example_2/assert_shapes/strided_slice_1/stack_1:output:0Fmodel_2/parse_example_2/assert_shapes/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
-model_2/parse_example_2/assert_shapes/Shape_2ShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:4*
T0	*
_output_shapes
::���
;model_2/parse_example_2/assert_shapes/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
=model_2/parse_example_2/assert_shapes/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=model_2/parse_example_2/assert_shapes/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5model_2/parse_example_2/assert_shapes/strided_slice_2StridedSlice6model_2/parse_example_2/assert_shapes/Shape_2:output:0Dmodel_2/parse_example_2/assert_shapes/strided_slice_2/stack:output:0Fmodel_2/parse_example_2/assert_shapes/strided_slice_2/stack_1:output:0Fmodel_2/parse_example_2/assert_shapes/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
-model_2/parse_example_2/assert_shapes/Shape_3ShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:5*
T0	*
_output_shapes
::���
;model_2/parse_example_2/assert_shapes/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: �
=model_2/parse_example_2/assert_shapes/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=model_2/parse_example_2/assert_shapes/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5model_2/parse_example_2/assert_shapes/strided_slice_3StridedSlice6model_2/parse_example_2/assert_shapes/Shape_3:output:0Dmodel_2/parse_example_2/assert_shapes/strided_slice_3/stack:output:0Fmodel_2/parse_example_2/assert_shapes/strided_slice_3/stack_1:output:0Fmodel_2/parse_example_2/assert_shapes/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
-model_2/parse_example_2/assert_shapes/Shape_4ShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:6*
T0	*
_output_shapes
::���
;model_2/parse_example_2/assert_shapes/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: �
=model_2/parse_example_2/assert_shapes/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=model_2/parse_example_2/assert_shapes/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5model_2/parse_example_2/assert_shapes/strided_slice_4StridedSlice6model_2/parse_example_2/assert_shapes/Shape_4:output:0Dmodel_2/parse_example_2/assert_shapes/strided_slice_4/stack:output:0Fmodel_2/parse_example_2/assert_shapes/strided_slice_4/stack_1:output:0Fmodel_2/parse_example_2/assert_shapes/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6model_2/parse_example_2/assert_shapes/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :�
7model_2/parse_example_2/assert_shapes/assert_rank/ShapeShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:2*
T0	*
_output_shapes
::��~
`model_2/parse_example_2/assert_shapes/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
Qmodel_2/parse_example_2/assert_shapes/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 z
8model_2/parse_example_2/assert_shapes/assert_rank_1/rankConst*
_output_shapes
: *
dtype0*
value	B :�
9model_2/parse_example_2/assert_shapes/assert_rank_1/ShapeShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:3*
T0	*
_output_shapes
::���
bmodel_2/parse_example_2/assert_shapes/assert_rank_1/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
Smodel_2/parse_example_2/assert_shapes/assert_rank_1/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 z
8model_2/parse_example_2/assert_shapes/assert_rank_2/rankConst*
_output_shapes
: *
dtype0*
value	B :�
9model_2/parse_example_2/assert_shapes/assert_rank_2/ShapeShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:4*
T0	*
_output_shapes
::���
bmodel_2/parse_example_2/assert_shapes/assert_rank_2/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
Smodel_2/parse_example_2/assert_shapes/assert_rank_2/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 z
8model_2/parse_example_2/assert_shapes/assert_rank_3/rankConst*
_output_shapes
: *
dtype0*
value	B :�
9model_2/parse_example_2/assert_shapes/assert_rank_3/ShapeShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:5*
T0	*
_output_shapes
::���
bmodel_2/parse_example_2/assert_shapes/assert_rank_3/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
Smodel_2/parse_example_2/assert_shapes/assert_rank_3/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 z
8model_2/parse_example_2/assert_shapes/assert_rank_4/rankConst*
_output_shapes
: *
dtype0*
value	B :�
9model_2/parse_example_2/assert_shapes/assert_rank_4/ShapeShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:6*
T0	*
_output_shapes
::���
bmodel_2/parse_example_2/assert_shapes/assert_rank_4/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 �
Smodel_2/parse_example_2/assert_shapes/assert_rank_4/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 �
+model_2/parse_example_2/assert_shapes/EqualEqual>model_2/parse_example_2/assert_shapes/strided_slice_1:output:0<model_2/parse_example_2/assert_shapes/strided_slice:output:0*
T0*
_output_shapes
: �
-model_2/parse_example_2/assert_shapes/Shape_5ShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:3*
T0	*
_output_shapes
::���
2model_2/parse_example_2/assert_shapes/Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLAll `#size` fields must have identical shapes for all node and edge sets..  �
4model_2/parse_example_2/assert_shapes/Assert/Const_1Const*
_output_shapes
: *
dtype0*f
value]B[ BUSpecified by tensor model_2/parse_example_2/ParseExample/ParseExampleV2:2 dimension 0�
4model_2/parse_example_2/assert_shapes/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFTensor model_2/parse_example_2/ParseExample/ParseExampleV2:3 dimensionv
4model_2/parse_example_2/assert_shapes/Assert/Const_3Const*
_output_shapes
: *
dtype0*
value	B : �
4model_2/parse_example_2/assert_shapes/Assert/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size�
4model_2/parse_example_2/assert_shapes/Assert/Const_5Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: �
:model_2/parse_example_2/assert_shapes/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLAll `#size` fields must have identical shapes for all node and edge sets..  �
:model_2/parse_example_2/assert_shapes/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*f
value]B[ BUSpecified by tensor model_2/parse_example_2/ParseExample/ParseExampleV2:2 dimension 0�
:model_2/parse_example_2/assert_shapes/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFTensor model_2/parse_example_2/ParseExample/ParseExampleV2:3 dimension|
:model_2/parse_example_2/assert_shapes/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*
value	B : �
:model_2/parse_example_2/assert_shapes/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size�
:model_2/parse_example_2/assert_shapes/Assert/Assert/data_6Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: �
3model_2/parse_example_2/assert_shapes/Assert/AssertAssert/model_2/parse_example_2/assert_shapes/Equal:z:0Cmodel_2/parse_example_2/assert_shapes/Assert/Assert/data_0:output:0Cmodel_2/parse_example_2/assert_shapes/Assert/Assert/data_1:output:0Cmodel_2/parse_example_2/assert_shapes/Assert/Assert/data_2:output:0Cmodel_2/parse_example_2/assert_shapes/Assert/Assert/data_3:output:0Cmodel_2/parse_example_2/assert_shapes/Assert/Assert/data_4:output:0<model_2/parse_example_2/assert_shapes/strided_slice:output:0Cmodel_2/parse_example_2/assert_shapes/Assert/Assert/data_6:output:06model_2/parse_example_2/assert_shapes/Shape_5:output:0K^model_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
-model_2/parse_example_2/assert_shapes/Equal_1Equal>model_2/parse_example_2/assert_shapes/strided_slice_2:output:0<model_2/parse_example_2/assert_shapes/strided_slice:output:0*
T0*
_output_shapes
: �
-model_2/parse_example_2/assert_shapes/Shape_6ShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:4*
T0	*
_output_shapes
::���
4model_2/parse_example_2/assert_shapes/Assert_1/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLAll `#size` fields must have identical shapes for all node and edge sets..  �
6model_2/parse_example_2/assert_shapes/Assert_1/Const_1Const*
_output_shapes
: *
dtype0*f
value]B[ BUSpecified by tensor model_2/parse_example_2/ParseExample/ParseExampleV2:2 dimension 0�
6model_2/parse_example_2/assert_shapes/Assert_1/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFTensor model_2/parse_example_2/ParseExample/ParseExampleV2:4 dimensionx
6model_2/parse_example_2/assert_shapes/Assert_1/Const_3Const*
_output_shapes
: *
dtype0*
value	B : �
6model_2/parse_example_2/assert_shapes/Assert_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size�
6model_2/parse_example_2/assert_shapes/Assert_1/Const_5Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: �
<model_2/parse_example_2/assert_shapes/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLAll `#size` fields must have identical shapes for all node and edge sets..  �
<model_2/parse_example_2/assert_shapes/Assert_1/Assert/data_1Const*
_output_shapes
: *
dtype0*f
value]B[ BUSpecified by tensor model_2/parse_example_2/ParseExample/ParseExampleV2:2 dimension 0�
<model_2/parse_example_2/assert_shapes/Assert_1/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFTensor model_2/parse_example_2/ParseExample/ParseExampleV2:4 dimension~
<model_2/parse_example_2/assert_shapes/Assert_1/Assert/data_3Const*
_output_shapes
: *
dtype0*
value	B : �
<model_2/parse_example_2/assert_shapes/Assert_1/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size�
<model_2/parse_example_2/assert_shapes/Assert_1/Assert/data_6Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: �
5model_2/parse_example_2/assert_shapes/Assert_1/AssertAssert1model_2/parse_example_2/assert_shapes/Equal_1:z:0Emodel_2/parse_example_2/assert_shapes/Assert_1/Assert/data_0:output:0Emodel_2/parse_example_2/assert_shapes/Assert_1/Assert/data_1:output:0Emodel_2/parse_example_2/assert_shapes/Assert_1/Assert/data_2:output:0Emodel_2/parse_example_2/assert_shapes/Assert_1/Assert/data_3:output:0Emodel_2/parse_example_2/assert_shapes/Assert_1/Assert/data_4:output:0<model_2/parse_example_2/assert_shapes/strided_slice:output:0Emodel_2/parse_example_2/assert_shapes/Assert_1/Assert/data_6:output:06model_2/parse_example_2/assert_shapes/Shape_6:output:04^model_2/parse_example_2/assert_shapes/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
-model_2/parse_example_2/assert_shapes/Equal_2Equal>model_2/parse_example_2/assert_shapes/strided_slice_3:output:0<model_2/parse_example_2/assert_shapes/strided_slice:output:0*
T0*
_output_shapes
: �
-model_2/parse_example_2/assert_shapes/Shape_7ShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:5*
T0	*
_output_shapes
::���
4model_2/parse_example_2/assert_shapes/Assert_2/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLAll `#size` fields must have identical shapes for all node and edge sets..  �
6model_2/parse_example_2/assert_shapes/Assert_2/Const_1Const*
_output_shapes
: *
dtype0*f
value]B[ BUSpecified by tensor model_2/parse_example_2/ParseExample/ParseExampleV2:2 dimension 0�
6model_2/parse_example_2/assert_shapes/Assert_2/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFTensor model_2/parse_example_2/ParseExample/ParseExampleV2:5 dimensionx
6model_2/parse_example_2/assert_shapes/Assert_2/Const_3Const*
_output_shapes
: *
dtype0*
value	B : �
6model_2/parse_example_2/assert_shapes/Assert_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size�
6model_2/parse_example_2/assert_shapes/Assert_2/Const_5Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: �
<model_2/parse_example_2/assert_shapes/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLAll `#size` fields must have identical shapes for all node and edge sets..  �
<model_2/parse_example_2/assert_shapes/Assert_2/Assert/data_1Const*
_output_shapes
: *
dtype0*f
value]B[ BUSpecified by tensor model_2/parse_example_2/ParseExample/ParseExampleV2:2 dimension 0�
<model_2/parse_example_2/assert_shapes/Assert_2/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFTensor model_2/parse_example_2/ParseExample/ParseExampleV2:5 dimension~
<model_2/parse_example_2/assert_shapes/Assert_2/Assert/data_3Const*
_output_shapes
: *
dtype0*
value	B : �
<model_2/parse_example_2/assert_shapes/Assert_2/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size�
<model_2/parse_example_2/assert_shapes/Assert_2/Assert/data_6Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: �
5model_2/parse_example_2/assert_shapes/Assert_2/AssertAssert1model_2/parse_example_2/assert_shapes/Equal_2:z:0Emodel_2/parse_example_2/assert_shapes/Assert_2/Assert/data_0:output:0Emodel_2/parse_example_2/assert_shapes/Assert_2/Assert/data_1:output:0Emodel_2/parse_example_2/assert_shapes/Assert_2/Assert/data_2:output:0Emodel_2/parse_example_2/assert_shapes/Assert_2/Assert/data_3:output:0Emodel_2/parse_example_2/assert_shapes/Assert_2/Assert/data_4:output:0<model_2/parse_example_2/assert_shapes/strided_slice:output:0Emodel_2/parse_example_2/assert_shapes/Assert_2/Assert/data_6:output:06model_2/parse_example_2/assert_shapes/Shape_7:output:06^model_2/parse_example_2/assert_shapes/Assert_1/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
-model_2/parse_example_2/assert_shapes/Equal_3Equal>model_2/parse_example_2/assert_shapes/strided_slice_4:output:0<model_2/parse_example_2/assert_shapes/strided_slice:output:0*
T0*
_output_shapes
: �
-model_2/parse_example_2/assert_shapes/Shape_8ShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:6*
T0	*
_output_shapes
::���
4model_2/parse_example_2/assert_shapes/Assert_3/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLAll `#size` fields must have identical shapes for all node and edge sets..  �
6model_2/parse_example_2/assert_shapes/Assert_3/Const_1Const*
_output_shapes
: *
dtype0*f
value]B[ BUSpecified by tensor model_2/parse_example_2/ParseExample/ParseExampleV2:2 dimension 0�
6model_2/parse_example_2/assert_shapes/Assert_3/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFTensor model_2/parse_example_2/ParseExample/ParseExampleV2:6 dimensionx
6model_2/parse_example_2/assert_shapes/Assert_3/Const_3Const*
_output_shapes
: *
dtype0*
value	B : �
6model_2/parse_example_2/assert_shapes/Assert_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size�
6model_2/parse_example_2/assert_shapes/Assert_3/Const_5Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: �
<model_2/parse_example_2/assert_shapes/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLAll `#size` fields must have identical shapes for all node and edge sets..  �
<model_2/parse_example_2/assert_shapes/Assert_3/Assert/data_1Const*
_output_shapes
: *
dtype0*f
value]B[ BUSpecified by tensor model_2/parse_example_2/ParseExample/ParseExampleV2:2 dimension 0�
<model_2/parse_example_2/assert_shapes/Assert_3/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFTensor model_2/parse_example_2/ParseExample/ParseExampleV2:6 dimension~
<model_2/parse_example_2/assert_shapes/Assert_3/Assert/data_3Const*
_output_shapes
: *
dtype0*
value	B : �
<model_2/parse_example_2/assert_shapes/Assert_3/Assert/data_4Const*
_output_shapes
: *
dtype0*
valueB Bmust have size�
<model_2/parse_example_2/assert_shapes/Assert_3/Assert/data_6Const*
_output_shapes
: *
dtype0*!
valueB BReceived shape: �
5model_2/parse_example_2/assert_shapes/Assert_3/AssertAssert1model_2/parse_example_2/assert_shapes/Equal_3:z:0Emodel_2/parse_example_2/assert_shapes/Assert_3/Assert/data_0:output:0Emodel_2/parse_example_2/assert_shapes/Assert_3/Assert/data_1:output:0Emodel_2/parse_example_2/assert_shapes/Assert_3/Assert/data_2:output:0Emodel_2/parse_example_2/assert_shapes/Assert_3/Assert/data_3:output:0Emodel_2/parse_example_2/assert_shapes/Assert_3/Assert/data_4:output:0<model_2/parse_example_2/assert_shapes/strided_slice:output:0Emodel_2/parse_example_2/assert_shapes/Assert_3/Assert/data_6:output:06model_2/parse_example_2/assert_shapes/Shape_8:output:06^model_2/parse_example_2/assert_shapes/Assert_2/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 �
"model_2/parse_example_2/group_depsNoOp4^model_2/parse_example_2/assert_shapes/Assert/Assert6^model_2/parse_example_2/assert_shapes/Assert_1/Assert6^model_2/parse_example_2/assert_shapes/Assert_2/Assert6^model_2/parse_example_2/assert_shapes/Assert_3/AssertR^model_2/parse_example_2/assert_shapes/assert_rank/static_checks_determined_all_okT^model_2/parse_example_2/assert_shapes/assert_rank_1/static_checks_determined_all_okT^model_2/parse_example_2/assert_shapes/assert_rank_2/static_checks_determined_all_okT^model_2/parse_example_2/assert_shapes/assert_rank_3/static_checks_determined_all_okT^model_2/parse_example_2/assert_shapes/assert_rank_4/static_checks_determined_all_ok*
_output_shapes
 �
model_2/parse_example_2/CastCastBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:2*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_2/parse_example_2/Cast_1CastBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:3*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_2/parse_example_2/Cast_2CastBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:4*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_2/parse_example_2/Cast_3CastBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:5*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_2/parse_example_2/Cast_4CastBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:6*

DstT0*

SrcT0	*'
_output_shapes
:����������
model_2/parse_example_2/Cast_5CastCmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
model_2/parse_example_2/Cast_6CastCmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:1*

DstT0*

SrcT0	*#
_output_shapes
:����������
model_2/parse_example_2/Cast_7CastCmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:2*

DstT0*

SrcT0	*#
_output_shapes
:����������
model_2/parse_example_2/Cast_8CastCmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:3*

DstT0*

SrcT0	*#
_output_shapes
:����������
model_2/parse_example_2/Cast_9CastDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:10*

DstT0*

SrcT0	*#
_output_shapes
:����������
model_2/parse_example_2/Cast_10CastDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:11*

DstT0*

SrcT0	*#
_output_shapes
:����������
model_2/parse_example_2/Shape_7ShapeBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:0*
T0	*
_output_shapes
::��w
-model_2/parse_example_2/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model_2/parse_example_2/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model_2/parse_example_2/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'model_2/parse_example_2/strided_slice_7StridedSlice(model_2/parse_example_2/Shape_7:output:06model_2/parse_example_2/strided_slice_7/stack:output:08model_2/parse_example_2/strided_slice_7/stack_1:output:08model_2/parse_example_2/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskd
"model_2/parse_example_2/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
model_2/parse_example_2/onesFill0model_2/parse_example_2/strided_slice_7:output:0+model_2/parse_example_2/ones/Const:output:0*
T0*'
_output_shapes
:����������
!model_2/parse_example_2/ones_likeOnesLike model_2/parse_example_2/Cast:y:0*
T0*'
_output_shapes
:����������
 model_2/parse_example_2/IdentityIdentityBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:0*
T0	*'
_output_shapes
:����������
"model_2/parse_example_2/Identity_1IdentityBmodel_2/parse_example_2/ParseExample/ParseExampleV2:dense_values:1*
T0*'
_output_shapes
:����������
"model_2/parse_example_2/Identity_2Identity%model_2/parse_example_2/ones_like:y:0*
T0*'
_output_shapes
:����������
"model_2/parse_example_2/Identity_3Identity"model_2/parse_example_2/Cast_5:y:0*
T0*#
_output_shapes
:����������
"model_2/parse_example_2/Identity_4IdentityGmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:0*
T0*#
_output_shapes
:����������
"model_2/parse_example_2/Identity_5Identity"model_2/parse_example_2/Cast_6:y:0*
T0*#
_output_shapes
:����������
"model_2/parse_example_2/Identity_6IdentityGmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:1*
T0*#
_output_shapes
:����������
"model_2/parse_example_2/Identity_7Identity model_2/parse_example_2/Cast:y:0*
T0*'
_output_shapes
:����������
"model_2/parse_example_2/Identity_8Identity"model_2/parse_example_2/Cast_7:y:0*
T0*#
_output_shapes
:����������
"model_2/parse_example_2/Identity_9IdentityGmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:2*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_10Identity"model_2/parse_example_2/Cast_8:y:0*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_11IdentityGmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:3*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_12Identity(model_2/parse_example_2/Reshape:output:0*
T0*'
_output_shapes
:����������
#model_2/parse_example_2/Identity_13IdentityGmodel_2/parse_example_2/RaggedFromRowSplits/control_dependency:output:0*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_14IdentityCmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:5*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_15IdentityGmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:5*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_16IdentityCmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:6*
T0	*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_17IdentityGmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:6*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_18IdentityCmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:7*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_19IdentityGmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:7*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_20IdentityCmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:8*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_21IdentityGmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:8*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_22Identity*model_2/parse_example_2/Reshape_1:output:0*
T0*'
_output_shapes
:����������
#model_2/parse_example_2/Identity_23IdentityImodel_2/parse_example_2/RaggedFromRowSplits_1/control_dependency:output:0*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_24Identity"model_2/parse_example_2/Cast_1:y:0*
T0*'
_output_shapes
:����������
#model_2/parse_example_2/Identity_25Identity"model_2/parse_example_2/Cast_9:y:0*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_26IdentityHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:10*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_27Identity#model_2/parse_example_2/Cast_10:y:0*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_28IdentityHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:11*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_29IdentityDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:12*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_30IdentityHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:12*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_31IdentityDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:13*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_32IdentityHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:13*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_33Identity"model_2/parse_example_2/Cast_2:y:0*
T0*'
_output_shapes
:����������
#model_2/parse_example_2/Identity_34Identity*model_2/parse_example_2/Reshape_3:output:0*
T0*+
_output_shapes
:����������
#model_2/parse_example_2/Identity_35IdentityImodel_2/parse_example_2/RaggedFromRowSplits_3/control_dependency:output:0*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_36Identity"model_2/parse_example_2/Cast_3:y:0*
T0*'
_output_shapes
:����������
#model_2/parse_example_2/Identity_37IdentityDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:15*
T0	*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_38IdentityHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:15*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_39Identity*model_2/parse_example_2/Reshape_4:output:0*
T0*'
_output_shapes
:����������
#model_2/parse_example_2/Identity_40IdentityImodel_2/parse_example_2/RaggedFromRowSplits_4/control_dependency:output:0*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_41Identity*model_2/parse_example_2/Reshape_5:output:0*
T0*'
_output_shapes
:���������	�
#model_2/parse_example_2/Identity_42IdentityImodel_2/parse_example_2/RaggedFromRowSplits_5/control_dependency:output:0*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_43IdentityDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:18*
T0	*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_44IdentityHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:18*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_45IdentityDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:19*
T0	*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_46IdentityHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:19*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_47Identity*model_2/parse_example_2/Reshape_6:output:0*
T0*'
_output_shapes
:���������	�
#model_2/parse_example_2/Identity_48IdentityImodel_2/parse_example_2/RaggedFromRowSplits_6/control_dependency:output:0*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_49IdentityDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:21*
T0	*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_50IdentityHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:21*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_51IdentityDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:22*
T0	*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_52IdentityHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:22*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_53IdentityDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:23*
T0	*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_54IdentityHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:23*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_55IdentityDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:24*
T0	*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_56IdentityHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:24*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_57IdentityDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:25*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_58IdentityHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:25*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_59IdentityDmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_values:26*
T0	*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_60IdentityHmodel_2/parse_example_2/ParseExample/ParseExampleV2:ragged_row_splits:26*
T0*#
_output_shapes
:����������
#model_2/parse_example_2/Identity_61Identity"model_2/parse_example_2/Cast_4:y:0*
T0*'
_output_shapes
:����������
Cmodel_2/model/input.merge_batch_to_components/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
1model_2/model/input.merge_batch_to_components/SumSum,model_2/parse_example_2/Identity_36:output:0Lmodel_2/model/input.merge_batch_to_components/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
;model_2/model/input.merge_batch_to_components/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
5model_2/model/input.merge_batch_to_components/ReshapeReshape:model_2/model/input.merge_batch_to_components/Sum:output:0Dmodel_2/model/input.merge_batch_to_components/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
Emodel_2/model/input.merge_batch_to_components/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
3model_2/model/input.merge_batch_to_components/Sum_1Sum,model_2/parse_example_2/Identity_61:output:0Nmodel_2/model/input.merge_batch_to_components/Sum_1/reduction_indices:output:0*
T0*#
_output_shapes
:����������
=model_2/model/input.merge_batch_to_components/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
7model_2/model/input.merge_batch_to_components/Reshape_1Reshape<model_2/model/input.merge_batch_to_components/Sum_1:output:0Fmodel_2/model/input.merge_batch_to_components/Reshape_1/shape:output:0*
T0*#
_output_shapes
:����������
3model_2/model/input.merge_batch_to_components/ShapeShape)model_2/parse_example_2/Identity:output:0*
T0	*
_output_shapes
::���
5model_2/model/input.merge_batch_to_components/unstackUnpack<model_2/model/input.merge_batch_to_components/Shape:output:0*
T0*
_output_shapes
: : *	
num�
=model_2/model/input.merge_batch_to_components/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
7model_2/model/input.merge_batch_to_components/Reshape_2Reshape)model_2/parse_example_2/Identity:output:0Fmodel_2/model/input.merge_batch_to_components/Reshape_2/shape:output:0*
T0	*#
_output_shapes
:����������
5model_2/model/input.merge_batch_to_components/Shape_1Shape+model_2/parse_example_2/Identity_1:output:0*
T0*
_output_shapes
::���
7model_2/model/input.merge_batch_to_components/unstack_1Unpack>model_2/model/input.merge_batch_to_components/Shape_1:output:0*
T0*
_output_shapes
: : *	
num�
=model_2/model/input.merge_batch_to_components/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
7model_2/model/input.merge_batch_to_components/Reshape_3Reshape+model_2/parse_example_2/Identity_1:output:0Fmodel_2/model/input.merge_batch_to_components/Reshape_3/shape:output:0*
T0*#
_output_shapes
:����������
5model_2/model/input.merge_batch_to_components/Shape_2Shape+model_2/parse_example_2/Identity_2:output:0*
T0*
_output_shapes
::���
7model_2/model/input.merge_batch_to_components/unstack_2Unpack>model_2/model/input.merge_batch_to_components/Shape_2:output:0*
T0*
_output_shapes
: : *	
num�
=model_2/model/input.merge_batch_to_components/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
7model_2/model/input.merge_batch_to_components/Reshape_4Reshape+model_2/parse_example_2/Identity_2:output:0Fmodel_2/model/input.merge_batch_to_components/Reshape_4/shape:output:0*
T0*#
_output_shapes
:����������
5model_2/model/input.merge_batch_to_components/Shape_3Shape,model_2/parse_example_2/Identity_36:output:0*
T0*
_output_shapes
::���
7model_2/model/input.merge_batch_to_components/unstack_3Unpack>model_2/model/input.merge_batch_to_components/Shape_3:output:0*
T0*
_output_shapes
: : *	
num�
=model_2/model/input.merge_batch_to_components/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
7model_2/model/input.merge_batch_to_components/Reshape_5Reshape,model_2/parse_example_2/Identity_36:output:0Fmodel_2/model/input.merge_batch_to_components/Reshape_5/shape:output:0*
T0*#
_output_shapes
:����������
5model_2/model/input.merge_batch_to_components/Shape_4Shape,model_2/parse_example_2/Identity_61:output:0*
T0*
_output_shapes
::���
7model_2/model/input.merge_batch_to_components/unstack_4Unpack>model_2/model/input.merge_batch_to_components/Shape_4:output:0*
T0*
_output_shapes
: : *	
num�
=model_2/model/input.merge_batch_to_components/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
7model_2/model/input.merge_batch_to_components/Reshape_6Reshape,model_2/parse_example_2/Identity_61:output:0Fmodel_2/model/input.merge_batch_to_components/Reshape_6/shape:output:0*
T0*#
_output_shapes
:����������
Emodel_2/model/input.merge_batch_to_components/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
3model_2/model/input.merge_batch_to_components/Sum_2Sum+model_2/parse_example_2/Identity_7:output:0Nmodel_2/model/input.merge_batch_to_components/Sum_2/reduction_indices:output:0*
T0*#
_output_shapes
:����������
=model_2/model/input.merge_batch_to_components/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
7model_2/model/input.merge_batch_to_components/Reshape_7Reshape<model_2/model/input.merge_batch_to_components/Sum_2:output:0Fmodel_2/model/input.merge_batch_to_components/Reshape_7/shape:output:0*
T0*#
_output_shapes
:����������
5model_2/model/input.merge_batch_to_components/Shape_5Shape+model_2/parse_example_2/Identity_3:output:0*
T0*
_output_shapes
::���
7model_2/model/input.merge_batch_to_components/unstack_5Unpack>model_2/model/input.merge_batch_to_components/Shape_5:output:0*
T0*
_output_shapes
: *	
num{
9model_2/model/input.merge_batch_to_components/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
4model_2/model/input.merge_batch_to_components/CumsumCumsum@model_2/model/input.merge_batch_to_components/Reshape_7:output:0Bmodel_2/model/input.merge_batch_to_components/Cumsum/axis:output:0*
T0*#
_output_shapes
:����������
7model_2/model/input.merge_batch_to_components/ones_likeOnesLike:model_2/model/input.merge_batch_to_components/Cumsum:out:0*
T0*#
_output_shapes
:���������u
3model_2/model/input.merge_batch_to_components/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
1model_2/model/input.merge_batch_to_components/addAddV2@model_2/model/input.merge_batch_to_components/unstack_5:output:0<model_2/model/input.merge_batch_to_components/add/y:output:0*
T0*
_output_shapes
: �
@model_2/model/input.merge_batch_to_components/UnsortedSegmentSumUnsortedSegmentSum;model_2/model/input.merge_batch_to_components/ones_like:y:0:model_2/model/input.merge_batch_to_components/Cumsum:out:05model_2/model/input.merge_batch_to_components/add:z:0*
Tindices0*
T0*#
_output_shapes
:���������}
;model_2/model/input.merge_batch_to_components/Cumsum_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6model_2/model/input.merge_batch_to_components/Cumsum_1CumsumImodel_2/model/input.merge_batch_to_components/UnsortedSegmentSum:output:0Dmodel_2/model/input.merge_batch_to_components/Cumsum_1/axis:output:0*
T0*#
_output_shapes
:���������u
3model_2/model/input.merge_batch_to_components/ConstConst*
_output_shapes
: *
dtype0*
value	B : w
5model_2/model/input.merge_batch_to_components/Const_1Const*
_output_shapes
: *
dtype0*
value	B :�
Amodel_2/model/input.merge_batch_to_components/strided_slice/stackPack<model_2/model/input.merge_batch_to_components/Const:output:0*
N*
T0*
_output_shapes
:�
Cmodel_2/model/input.merge_batch_to_components/strided_slice/stack_1Pack@model_2/model/input.merge_batch_to_components/unstack_5:output:0*
N*
T0*
_output_shapes
:�
Cmodel_2/model/input.merge_batch_to_components/strided_slice/stack_2Pack>model_2/model/input.merge_batch_to_components/Const_1:output:0*
N*
T0*
_output_shapes
:�
;model_2/model/input.merge_batch_to_components/strided_sliceStridedSlice<model_2/model/input.merge_batch_to_components/Cumsum_1:out:0Jmodel_2/model/input.merge_batch_to_components/strided_slice/stack:output:0Lmodel_2/model/input.merge_batch_to_components/strided_slice/stack_1:output:0Lmodel_2/model/input.merge_batch_to_components/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask}
;model_2/model/input.merge_batch_to_components/Cumsum_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6model_2/model/input.merge_batch_to_components/Cumsum_2Cumsum@model_2/model/input.merge_batch_to_components/Reshape_1:output:0Dmodel_2/model/input.merge_batch_to_components/Cumsum_2/axis:output:0*
T0*#
_output_shapes
:���������*
	exclusive(}
;model_2/model/input.merge_batch_to_components/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6model_2/model/input.merge_batch_to_components/GatherV2GatherV2<model_2/model/input.merge_batch_to_components/Cumsum_2:out:0Dmodel_2/model/input.merge_batch_to_components/strided_slice:output:0Dmodel_2/model/input.merge_batch_to_components/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:����������
3model_2/model/input.merge_batch_to_components/Add_1AddV2+model_2/parse_example_2/Identity_3:output:0?model_2/model/input.merge_batch_to_components/GatherV2:output:0*
T0*#
_output_shapes
:����������
5model_2/model/input.merge_batch_to_components/Shape_6Shape+model_2/parse_example_2/Identity_5:output:0*
T0*
_output_shapes
::���
7model_2/model/input.merge_batch_to_components/unstack_6Unpack>model_2/model/input.merge_batch_to_components/Shape_6:output:0*
T0*
_output_shapes
: *	
num}
;model_2/model/input.merge_batch_to_components/Cumsum_3/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6model_2/model/input.merge_batch_to_components/Cumsum_3Cumsum@model_2/model/input.merge_batch_to_components/Reshape_7:output:0Dmodel_2/model/input.merge_batch_to_components/Cumsum_3/axis:output:0*
T0*#
_output_shapes
:����������
9model_2/model/input.merge_batch_to_components/ones_like_1OnesLike<model_2/model/input.merge_batch_to_components/Cumsum_3:out:0*
T0*#
_output_shapes
:���������w
5model_2/model/input.merge_batch_to_components/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
3model_2/model/input.merge_batch_to_components/add_2AddV2@model_2/model/input.merge_batch_to_components/unstack_6:output:0>model_2/model/input.merge_batch_to_components/add_2/y:output:0*
T0*
_output_shapes
: �
Bmodel_2/model/input.merge_batch_to_components/UnsortedSegmentSum_1UnsortedSegmentSum=model_2/model/input.merge_batch_to_components/ones_like_1:y:0<model_2/model/input.merge_batch_to_components/Cumsum_3:out:07model_2/model/input.merge_batch_to_components/add_2:z:0*
Tindices0*
T0*#
_output_shapes
:���������}
;model_2/model/input.merge_batch_to_components/Cumsum_4/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6model_2/model/input.merge_batch_to_components/Cumsum_4CumsumKmodel_2/model/input.merge_batch_to_components/UnsortedSegmentSum_1:output:0Dmodel_2/model/input.merge_batch_to_components/Cumsum_4/axis:output:0*
T0*#
_output_shapes
:���������w
5model_2/model/input.merge_batch_to_components/Const_2Const*
_output_shapes
: *
dtype0*
value	B : w
5model_2/model/input.merge_batch_to_components/Const_3Const*
_output_shapes
: *
dtype0*
value	B :�
Cmodel_2/model/input.merge_batch_to_components/strided_slice_1/stackPack>model_2/model/input.merge_batch_to_components/Const_2:output:0*
N*
T0*
_output_shapes
:�
Emodel_2/model/input.merge_batch_to_components/strided_slice_1/stack_1Pack@model_2/model/input.merge_batch_to_components/unstack_6:output:0*
N*
T0*
_output_shapes
:�
Emodel_2/model/input.merge_batch_to_components/strided_slice_1/stack_2Pack>model_2/model/input.merge_batch_to_components/Const_3:output:0*
N*
T0*
_output_shapes
:�
=model_2/model/input.merge_batch_to_components/strided_slice_1StridedSlice<model_2/model/input.merge_batch_to_components/Cumsum_4:out:0Lmodel_2/model/input.merge_batch_to_components/strided_slice_1/stack:output:0Nmodel_2/model/input.merge_batch_to_components/strided_slice_1/stack_1:output:0Nmodel_2/model/input.merge_batch_to_components/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask}
;model_2/model/input.merge_batch_to_components/Cumsum_5/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6model_2/model/input.merge_batch_to_components/Cumsum_5Cumsum>model_2/model/input.merge_batch_to_components/Reshape:output:0Dmodel_2/model/input.merge_batch_to_components/Cumsum_5/axis:output:0*
T0*#
_output_shapes
:���������*
	exclusive(
=model_2/model/input.merge_batch_to_components/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8model_2/model/input.merge_batch_to_components/GatherV2_1GatherV2<model_2/model/input.merge_batch_to_components/Cumsum_5:out:0Fmodel_2/model/input.merge_batch_to_components/strided_slice_1:output:0Fmodel_2/model/input.merge_batch_to_components/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:����������
3model_2/model/input.merge_batch_to_components/Add_3AddV2+model_2/parse_example_2/Identity_5:output:0Amodel_2/model/input.merge_batch_to_components/GatherV2_1:output:0*
T0*#
_output_shapes
:����������
5model_2/model/input.merge_batch_to_components/Shape_7Shape+model_2/parse_example_2/Identity_7:output:0*
T0*
_output_shapes
::���
7model_2/model/input.merge_batch_to_components/unstack_7Unpack>model_2/model/input.merge_batch_to_components/Shape_7:output:0*
T0*
_output_shapes
: : *	
num�
=model_2/model/input.merge_batch_to_components/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
7model_2/model/input.merge_batch_to_components/Reshape_8Reshape+model_2/parse_example_2/Identity_7:output:0Fmodel_2/model/input.merge_batch_to_components/Reshape_8/shape:output:0*
T0*#
_output_shapes
:����������
Emodel_2/model/input.merge_batch_to_components/Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
3model_2/model/input.merge_batch_to_components/Sum_3Sum,model_2/parse_example_2/Identity_24:output:0Nmodel_2/model/input.merge_batch_to_components/Sum_3/reduction_indices:output:0*
T0*#
_output_shapes
:����������
=model_2/model/input.merge_batch_to_components/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
7model_2/model/input.merge_batch_to_components/Reshape_9Reshape<model_2/model/input.merge_batch_to_components/Sum_3:output:0Fmodel_2/model/input.merge_batch_to_components/Reshape_9/shape:output:0*
T0*#
_output_shapes
:����������
5model_2/model/input.merge_batch_to_components/Shape_8Shape+model_2/parse_example_2/Identity_8:output:0*
T0*
_output_shapes
::���
7model_2/model/input.merge_batch_to_components/unstack_8Unpack>model_2/model/input.merge_batch_to_components/Shape_8:output:0*
T0*
_output_shapes
: *	
num}
;model_2/model/input.merge_batch_to_components/Cumsum_6/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6model_2/model/input.merge_batch_to_components/Cumsum_6Cumsum@model_2/model/input.merge_batch_to_components/Reshape_9:output:0Dmodel_2/model/input.merge_batch_to_components/Cumsum_6/axis:output:0*
T0*#
_output_shapes
:����������
9model_2/model/input.merge_batch_to_components/ones_like_2OnesLike<model_2/model/input.merge_batch_to_components/Cumsum_6:out:0*
T0*#
_output_shapes
:���������w
5model_2/model/input.merge_batch_to_components/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :�
3model_2/model/input.merge_batch_to_components/add_4AddV2@model_2/model/input.merge_batch_to_components/unstack_8:output:0>model_2/model/input.merge_batch_to_components/add_4/y:output:0*
T0*
_output_shapes
: �
Bmodel_2/model/input.merge_batch_to_components/UnsortedSegmentSum_2UnsortedSegmentSum=model_2/model/input.merge_batch_to_components/ones_like_2:y:0<model_2/model/input.merge_batch_to_components/Cumsum_6:out:07model_2/model/input.merge_batch_to_components/add_4:z:0*
Tindices0*
T0*#
_output_shapes
:���������}
;model_2/model/input.merge_batch_to_components/Cumsum_7/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6model_2/model/input.merge_batch_to_components/Cumsum_7CumsumKmodel_2/model/input.merge_batch_to_components/UnsortedSegmentSum_2:output:0Dmodel_2/model/input.merge_batch_to_components/Cumsum_7/axis:output:0*
T0*#
_output_shapes
:���������w
5model_2/model/input.merge_batch_to_components/Const_4Const*
_output_shapes
: *
dtype0*
value	B : w
5model_2/model/input.merge_batch_to_components/Const_5Const*
_output_shapes
: *
dtype0*
value	B :�
Cmodel_2/model/input.merge_batch_to_components/strided_slice_2/stackPack>model_2/model/input.merge_batch_to_components/Const_4:output:0*
N*
T0*
_output_shapes
:�
Emodel_2/model/input.merge_batch_to_components/strided_slice_2/stack_1Pack@model_2/model/input.merge_batch_to_components/unstack_8:output:0*
N*
T0*
_output_shapes
:�
Emodel_2/model/input.merge_batch_to_components/strided_slice_2/stack_2Pack>model_2/model/input.merge_batch_to_components/Const_5:output:0*
N*
T0*
_output_shapes
:�
=model_2/model/input.merge_batch_to_components/strided_slice_2StridedSlice<model_2/model/input.merge_batch_to_components/Cumsum_7:out:0Lmodel_2/model/input.merge_batch_to_components/strided_slice_2/stack:output:0Nmodel_2/model/input.merge_batch_to_components/strided_slice_2/stack_1:output:0Nmodel_2/model/input.merge_batch_to_components/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask}
;model_2/model/input.merge_batch_to_components/Cumsum_8/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6model_2/model/input.merge_batch_to_components/Cumsum_8Cumsum@model_2/model/input.merge_batch_to_components/Reshape_1:output:0Dmodel_2/model/input.merge_batch_to_components/Cumsum_8/axis:output:0*
T0*#
_output_shapes
:���������*
	exclusive(
=model_2/model/input.merge_batch_to_components/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8model_2/model/input.merge_batch_to_components/GatherV2_2GatherV2<model_2/model/input.merge_batch_to_components/Cumsum_8:out:0Fmodel_2/model/input.merge_batch_to_components/strided_slice_2:output:0Fmodel_2/model/input.merge_batch_to_components/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:����������
3model_2/model/input.merge_batch_to_components/Add_5AddV2+model_2/parse_example_2/Identity_8:output:0Amodel_2/model/input.merge_batch_to_components/GatherV2_2:output:0*
T0*#
_output_shapes
:����������
5model_2/model/input.merge_batch_to_components/Shape_9Shape,model_2/parse_example_2/Identity_10:output:0*
T0*
_output_shapes
::���
7model_2/model/input.merge_batch_to_components/unstack_9Unpack>model_2/model/input.merge_batch_to_components/Shape_9:output:0*
T0*
_output_shapes
: *	
num}
;model_2/model/input.merge_batch_to_components/Cumsum_9/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6model_2/model/input.merge_batch_to_components/Cumsum_9Cumsum@model_2/model/input.merge_batch_to_components/Reshape_9:output:0Dmodel_2/model/input.merge_batch_to_components/Cumsum_9/axis:output:0*
T0*#
_output_shapes
:����������
9model_2/model/input.merge_batch_to_components/ones_like_3OnesLike<model_2/model/input.merge_batch_to_components/Cumsum_9:out:0*
T0*#
_output_shapes
:���������w
5model_2/model/input.merge_batch_to_components/add_6/yConst*
_output_shapes
: *
dtype0*
value	B :�
3model_2/model/input.merge_batch_to_components/add_6AddV2@model_2/model/input.merge_batch_to_components/unstack_9:output:0>model_2/model/input.merge_batch_to_components/add_6/y:output:0*
T0*
_output_shapes
: �
Bmodel_2/model/input.merge_batch_to_components/UnsortedSegmentSum_3UnsortedSegmentSum=model_2/model/input.merge_batch_to_components/ones_like_3:y:0<model_2/model/input.merge_batch_to_components/Cumsum_9:out:07model_2/model/input.merge_batch_to_components/add_6:z:0*
Tindices0*
T0*#
_output_shapes
:���������~
<model_2/model/input.merge_batch_to_components/Cumsum_10/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7model_2/model/input.merge_batch_to_components/Cumsum_10CumsumKmodel_2/model/input.merge_batch_to_components/UnsortedSegmentSum_3:output:0Emodel_2/model/input.merge_batch_to_components/Cumsum_10/axis:output:0*
T0*#
_output_shapes
:���������w
5model_2/model/input.merge_batch_to_components/Const_6Const*
_output_shapes
: *
dtype0*
value	B : w
5model_2/model/input.merge_batch_to_components/Const_7Const*
_output_shapes
: *
dtype0*
value	B :�
Cmodel_2/model/input.merge_batch_to_components/strided_slice_3/stackPack>model_2/model/input.merge_batch_to_components/Const_6:output:0*
N*
T0*
_output_shapes
:�
Emodel_2/model/input.merge_batch_to_components/strided_slice_3/stack_1Pack@model_2/model/input.merge_batch_to_components/unstack_9:output:0*
N*
T0*
_output_shapes
:�
Emodel_2/model/input.merge_batch_to_components/strided_slice_3/stack_2Pack>model_2/model/input.merge_batch_to_components/Const_7:output:0*
N*
T0*
_output_shapes
:�
=model_2/model/input.merge_batch_to_components/strided_slice_3StridedSlice=model_2/model/input.merge_batch_to_components/Cumsum_10:out:0Lmodel_2/model/input.merge_batch_to_components/strided_slice_3/stack:output:0Nmodel_2/model/input.merge_batch_to_components/strided_slice_3/stack_1:output:0Nmodel_2/model/input.merge_batch_to_components/strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask~
<model_2/model/input.merge_batch_to_components/Cumsum_11/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7model_2/model/input.merge_batch_to_components/Cumsum_11Cumsum@model_2/model/input.merge_batch_to_components/Reshape_1:output:0Emodel_2/model/input.merge_batch_to_components/Cumsum_11/axis:output:0*
T0*#
_output_shapes
:���������*
	exclusive(
=model_2/model/input.merge_batch_to_components/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8model_2/model/input.merge_batch_to_components/GatherV2_3GatherV2=model_2/model/input.merge_batch_to_components/Cumsum_11:out:0Fmodel_2/model/input.merge_batch_to_components/strided_slice_3:output:0Fmodel_2/model/input.merge_batch_to_components/GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:����������
3model_2/model/input.merge_batch_to_components/Add_7AddV2,model_2/parse_example_2/Identity_10:output:0Amodel_2/model/input.merge_batch_to_components/GatherV2_3:output:0*
T0*#
_output_shapes
:����������
6model_2/model/input.merge_batch_to_components/Shape_10Shape,model_2/parse_example_2/Identity_24:output:0*
T0*
_output_shapes
::���
8model_2/model/input.merge_batch_to_components/unstack_10Unpack?model_2/model/input.merge_batch_to_components/Shape_10:output:0*
T0*
_output_shapes
: : *	
num�
>model_2/model/input.merge_batch_to_components/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
8model_2/model/input.merge_batch_to_components/Reshape_10Reshape,model_2/parse_example_2/Identity_24:output:0Gmodel_2/model/input.merge_batch_to_components/Reshape_10/shape:output:0*
T0*#
_output_shapes
:����������
Emodel_2/model/input.merge_batch_to_components/Sum_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
3model_2/model/input.merge_batch_to_components/Sum_4Sum,model_2/parse_example_2/Identity_33:output:0Nmodel_2/model/input.merge_batch_to_components/Sum_4/reduction_indices:output:0*
T0*#
_output_shapes
:����������
>model_2/model/input.merge_batch_to_components/Reshape_11/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
8model_2/model/input.merge_batch_to_components/Reshape_11Reshape<model_2/model/input.merge_batch_to_components/Sum_4:output:0Gmodel_2/model/input.merge_batch_to_components/Reshape_11/shape:output:0*
T0*#
_output_shapes
:����������
6model_2/model/input.merge_batch_to_components/Shape_11Shape,model_2/parse_example_2/Identity_25:output:0*
T0*
_output_shapes
::���
8model_2/model/input.merge_batch_to_components/unstack_11Unpack?model_2/model/input.merge_batch_to_components/Shape_11:output:0*
T0*
_output_shapes
: *	
num~
<model_2/model/input.merge_batch_to_components/Cumsum_12/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7model_2/model/input.merge_batch_to_components/Cumsum_12CumsumAmodel_2/model/input.merge_batch_to_components/Reshape_11:output:0Emodel_2/model/input.merge_batch_to_components/Cumsum_12/axis:output:0*
T0*#
_output_shapes
:����������
9model_2/model/input.merge_batch_to_components/ones_like_4OnesLike=model_2/model/input.merge_batch_to_components/Cumsum_12:out:0*
T0*#
_output_shapes
:���������w
5model_2/model/input.merge_batch_to_components/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :�
3model_2/model/input.merge_batch_to_components/add_8AddV2Amodel_2/model/input.merge_batch_to_components/unstack_11:output:0>model_2/model/input.merge_batch_to_components/add_8/y:output:0*
T0*
_output_shapes
: �
Bmodel_2/model/input.merge_batch_to_components/UnsortedSegmentSum_4UnsortedSegmentSum=model_2/model/input.merge_batch_to_components/ones_like_4:y:0=model_2/model/input.merge_batch_to_components/Cumsum_12:out:07model_2/model/input.merge_batch_to_components/add_8:z:0*
Tindices0*
T0*#
_output_shapes
:���������~
<model_2/model/input.merge_batch_to_components/Cumsum_13/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7model_2/model/input.merge_batch_to_components/Cumsum_13CumsumKmodel_2/model/input.merge_batch_to_components/UnsortedSegmentSum_4:output:0Emodel_2/model/input.merge_batch_to_components/Cumsum_13/axis:output:0*
T0*#
_output_shapes
:���������w
5model_2/model/input.merge_batch_to_components/Const_8Const*
_output_shapes
: *
dtype0*
value	B : w
5model_2/model/input.merge_batch_to_components/Const_9Const*
_output_shapes
: *
dtype0*
value	B :�
Cmodel_2/model/input.merge_batch_to_components/strided_slice_4/stackPack>model_2/model/input.merge_batch_to_components/Const_8:output:0*
N*
T0*
_output_shapes
:�
Emodel_2/model/input.merge_batch_to_components/strided_slice_4/stack_1PackAmodel_2/model/input.merge_batch_to_components/unstack_11:output:0*
N*
T0*
_output_shapes
:�
Emodel_2/model/input.merge_batch_to_components/strided_slice_4/stack_2Pack>model_2/model/input.merge_batch_to_components/Const_9:output:0*
N*
T0*
_output_shapes
:�
=model_2/model/input.merge_batch_to_components/strided_slice_4StridedSlice=model_2/model/input.merge_batch_to_components/Cumsum_13:out:0Lmodel_2/model/input.merge_batch_to_components/strided_slice_4/stack:output:0Nmodel_2/model/input.merge_batch_to_components/strided_slice_4/stack_1:output:0Nmodel_2/model/input.merge_batch_to_components/strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask~
<model_2/model/input.merge_batch_to_components/Cumsum_14/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7model_2/model/input.merge_batch_to_components/Cumsum_14Cumsum@model_2/model/input.merge_batch_to_components/Reshape_1:output:0Emodel_2/model/input.merge_batch_to_components/Cumsum_14/axis:output:0*
T0*#
_output_shapes
:���������*
	exclusive(
=model_2/model/input.merge_batch_to_components/GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8model_2/model/input.merge_batch_to_components/GatherV2_4GatherV2=model_2/model/input.merge_batch_to_components/Cumsum_14:out:0Fmodel_2/model/input.merge_batch_to_components/strided_slice_4:output:0Fmodel_2/model/input.merge_batch_to_components/GatherV2_4/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:����������
3model_2/model/input.merge_batch_to_components/Add_9AddV2,model_2/parse_example_2/Identity_25:output:0Amodel_2/model/input.merge_batch_to_components/GatherV2_4:output:0*
T0*#
_output_shapes
:����������
6model_2/model/input.merge_batch_to_components/Shape_12Shape,model_2/parse_example_2/Identity_27:output:0*
T0*
_output_shapes
::���
8model_2/model/input.merge_batch_to_components/unstack_12Unpack?model_2/model/input.merge_batch_to_components/Shape_12:output:0*
T0*
_output_shapes
: *	
num~
<model_2/model/input.merge_batch_to_components/Cumsum_15/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7model_2/model/input.merge_batch_to_components/Cumsum_15CumsumAmodel_2/model/input.merge_batch_to_components/Reshape_11:output:0Emodel_2/model/input.merge_batch_to_components/Cumsum_15/axis:output:0*
T0*#
_output_shapes
:����������
9model_2/model/input.merge_batch_to_components/ones_like_5OnesLike=model_2/model/input.merge_batch_to_components/Cumsum_15:out:0*
T0*#
_output_shapes
:���������x
6model_2/model/input.merge_batch_to_components/add_10/yConst*
_output_shapes
: *
dtype0*
value	B :�
4model_2/model/input.merge_batch_to_components/add_10AddV2Amodel_2/model/input.merge_batch_to_components/unstack_12:output:0?model_2/model/input.merge_batch_to_components/add_10/y:output:0*
T0*
_output_shapes
: �
Bmodel_2/model/input.merge_batch_to_components/UnsortedSegmentSum_5UnsortedSegmentSum=model_2/model/input.merge_batch_to_components/ones_like_5:y:0=model_2/model/input.merge_batch_to_components/Cumsum_15:out:08model_2/model/input.merge_batch_to_components/add_10:z:0*
Tindices0*
T0*#
_output_shapes
:���������~
<model_2/model/input.merge_batch_to_components/Cumsum_16/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7model_2/model/input.merge_batch_to_components/Cumsum_16CumsumKmodel_2/model/input.merge_batch_to_components/UnsortedSegmentSum_5:output:0Emodel_2/model/input.merge_batch_to_components/Cumsum_16/axis:output:0*
T0*#
_output_shapes
:���������x
6model_2/model/input.merge_batch_to_components/Const_10Const*
_output_shapes
: *
dtype0*
value	B : x
6model_2/model/input.merge_batch_to_components/Const_11Const*
_output_shapes
: *
dtype0*
value	B :�
Cmodel_2/model/input.merge_batch_to_components/strided_slice_5/stackPack?model_2/model/input.merge_batch_to_components/Const_10:output:0*
N*
T0*
_output_shapes
:�
Emodel_2/model/input.merge_batch_to_components/strided_slice_5/stack_1PackAmodel_2/model/input.merge_batch_to_components/unstack_12:output:0*
N*
T0*
_output_shapes
:�
Emodel_2/model/input.merge_batch_to_components/strided_slice_5/stack_2Pack?model_2/model/input.merge_batch_to_components/Const_11:output:0*
N*
T0*
_output_shapes
:�
=model_2/model/input.merge_batch_to_components/strided_slice_5StridedSlice=model_2/model/input.merge_batch_to_components/Cumsum_16:out:0Lmodel_2/model/input.merge_batch_to_components/strided_slice_5/stack:output:0Nmodel_2/model/input.merge_batch_to_components/strided_slice_5/stack_1:output:0Nmodel_2/model/input.merge_batch_to_components/strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask~
<model_2/model/input.merge_batch_to_components/Cumsum_17/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7model_2/model/input.merge_batch_to_components/Cumsum_17Cumsum@model_2/model/input.merge_batch_to_components/Reshape_1:output:0Emodel_2/model/input.merge_batch_to_components/Cumsum_17/axis:output:0*
T0*#
_output_shapes
:���������*
	exclusive(
=model_2/model/input.merge_batch_to_components/GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8model_2/model/input.merge_batch_to_components/GatherV2_5GatherV2=model_2/model/input.merge_batch_to_components/Cumsum_17:out:0Fmodel_2/model/input.merge_batch_to_components/strided_slice_5:output:0Fmodel_2/model/input.merge_batch_to_components/GatherV2_5/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:����������
4model_2/model/input.merge_batch_to_components/Add_11AddV2,model_2/parse_example_2/Identity_27:output:0Amodel_2/model/input.merge_batch_to_components/GatherV2_5:output:0*
T0*#
_output_shapes
:����������
6model_2/model/input.merge_batch_to_components/Shape_13Shape,model_2/parse_example_2/Identity_33:output:0*
T0*
_output_shapes
::���
8model_2/model/input.merge_batch_to_components/unstack_13Unpack?model_2/model/input.merge_batch_to_components/Shape_13:output:0*
T0*
_output_shapes
: : *	
num�
>model_2/model/input.merge_batch_to_components/Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
8model_2/model/input.merge_batch_to_components/Reshape_12Reshape,model_2/parse_example_2/Identity_33:output:0Gmodel_2/model/input.merge_batch_to_components/Reshape_12/shape:output:0*
T0*#
_output_shapes
:����������
9model_2/model/input.merge_batch_to_components/ones_like_6OnesLike@model_2/model/input.merge_batch_to_components/Reshape_8:output:0*
T0*#
_output_shapes
:����������
model_2/model/tf.cast/CastCast,model_2/parse_example_2/Identity_34:output:0*

DstT0*

SrcT0*+
_output_shapes
:����������
-model_2/model/input.remove_features/ones_likeOnesLike@model_2/model/input.merge_batch_to_components/Reshape_8:output:0*
T0*#
_output_shapes
:����������
Kmodel_2/model_1/map_features/model/atom_sym_embedding/MatMul/ReadVariableOpReadVariableOpTmodel_2_model_1_map_features_model_atom_sym_embedding_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
<model_2/model_1/map_features/model/atom_sym_embedding/MatMulMatMul,model_2/parse_example_2/Identity_39:output:0Smodel_2/model_1/map_features/model/atom_sym_embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Lmodel_2/model_1/map_features/model/atom_sym_embedding/BiasAdd/ReadVariableOpReadVariableOpUmodel_2_model_1_map_features_model_atom_sym_embedding_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
=model_2/model_1/map_features/model/atom_sym_embedding/BiasAddBiasAddFmodel_2/model_1/map_features/model/atom_sym_embedding/MatMul:product:0Tmodel_2/model_1/map_features/model/atom_sym_embedding/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Mmodel_2/model_1/map_features/model/chiral_tag_embedding/MatMul/ReadVariableOpReadVariableOpVmodel_2_model_1_map_features_model_chiral_tag_embedding_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0�
>model_2/model_1/map_features/model/chiral_tag_embedding/MatMulMatMul,model_2/parse_example_2/Identity_41:output:0Umodel_2/model_1/map_features/model/chiral_tag_embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Nmodel_2/model_1/map_features/model/chiral_tag_embedding/BiasAdd/ReadVariableOpReadVariableOpWmodel_2_model_1_map_features_model_chiral_tag_embedding_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
?model_2/model_1/map_features/model/chiral_tag_embedding/BiasAddBiasAddHmodel_2/model_1/map_features/model/chiral_tag_embedding/MatMul:product:0Vmodel_2/model_1/map_features/model/chiral_tag_embedding/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Pmodel_2/model_1/map_features/model/hybridization_embedding/MatMul/ReadVariableOpReadVariableOpYmodel_2_model_1_map_features_model_hybridization_embedding_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0�
Amodel_2/model_1/map_features/model/hybridization_embedding/MatMulMatMul,model_2/parse_example_2/Identity_47:output:0Xmodel_2/model_1/map_features/model/hybridization_embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Qmodel_2/model_1/map_features/model/hybridization_embedding/BiasAdd/ReadVariableOpReadVariableOpZmodel_2_model_1_map_features_model_hybridization_embedding_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Bmodel_2/model_1/map_features/model/hybridization_embedding/BiasAddBiasAddKmodel_2/model_1/map_features/model/hybridization_embedding/MatMul:product:0Ymodel_2/model_1/map_features/model/hybridization_embedding/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Dmodel_2/model_1/map_features/model/degree_embedding/embedding_lookupResourceGatherKmodel_2_model_1_map_features_model_degree_embedding_embedding_lookup_315371,model_2/parse_example_2/Identity_43:output:0*
Tindices0	*^
_classT
RPloc:@model_2/model_1/map_features/model/degree_embedding/embedding_lookup/315371*'
_output_shapes
:���������*
dtype0�
Mmodel_2/model_1/map_features/model/degree_embedding/embedding_lookup/IdentityIdentityMmodel_2/model_1/map_features/model/degree_embedding/embedding_lookup:output:0*
T0*'
_output_shapes
:����������
Kmodel_2/model_1/map_features/model/formal_charge_embedding/embedding_lookupResourceGatherRmodel_2_model_1_map_features_model_formal_charge_embedding_embedding_lookup_315375,model_2/parse_example_2/Identity_45:output:0*
Tindices0	*e
_class[
YWloc:@model_2/model_1/map_features/model/formal_charge_embedding/embedding_lookup/315375*'
_output_shapes
:���������*
dtype0�
Tmodel_2/model_1/map_features/model/formal_charge_embedding/embedding_lookup/IdentityIdentityTmodel_2/model_1/map_features/model/formal_charge_embedding/embedding_lookup:output:0*
T0*'
_output_shapes
:����������
Imodel_2/model_1/map_features/model/is_aromatic_embedding/embedding_lookupResourceGatherPmodel_2_model_1_map_features_model_is_aromatic_embedding_embedding_lookup_315379,model_2/parse_example_2/Identity_49:output:0*
Tindices0	*c
_classY
WUloc:@model_2/model_1/map_features/model/is_aromatic_embedding/embedding_lookup/315379*'
_output_shapes
:���������*
dtype0�
Rmodel_2/model_1/map_features/model/is_aromatic_embedding/embedding_lookup/IdentityIdentityRmodel_2/model_1/map_features/model/is_aromatic_embedding/embedding_lookup:output:0*
T0*'
_output_shapes
:����������
Imodel_2/model_1/map_features/model/no_implicit_embedding/embedding_lookupResourceGatherPmodel_2_model_1_map_features_model_no_implicit_embedding_embedding_lookup_315383,model_2/parse_example_2/Identity_51:output:0*
Tindices0	*c
_classY
WUloc:@model_2/model_1/map_features/model/no_implicit_embedding/embedding_lookup/315383*'
_output_shapes
:���������*
dtype0�
Rmodel_2/model_1/map_features/model/no_implicit_embedding/embedding_lookup/IdentityIdentityRmodel_2/model_1/map_features/model/no_implicit_embedding/embedding_lookup:output:0*
T0*'
_output_shapes
:����������
Dmodel_2/model_1/map_features/model/num_Hs_embedding/embedding_lookupResourceGatherKmodel_2_model_1_map_features_model_num_hs_embedding_embedding_lookup_315387,model_2/parse_example_2/Identity_53:output:0*
Tindices0	*^
_classT
RPloc:@model_2/model_1/map_features/model/num_Hs_embedding/embedding_lookup/315387*'
_output_shapes
:���������*
dtype0�
Mmodel_2/model_1/map_features/model/num_Hs_embedding/embedding_lookup/IdentityIdentityMmodel_2/model_1/map_features/model/num_Hs_embedding/embedding_lookup:output:0*
T0*'
_output_shapes
:����������
Emodel_2/model_1/map_features/model/valence_embedding/embedding_lookupResourceGatherLmodel_2_model_1_map_features_model_valence_embedding_embedding_lookup_315391,model_2/parse_example_2/Identity_59:output:0*
Tindices0	*_
_classU
SQloc:@model_2/model_1/map_features/model/valence_embedding/embedding_lookup/315391*'
_output_shapes
:���������*
dtype0�
Nmodel_2/model_1/map_features/model/valence_embedding/embedding_lookup/IdentityIdentityNmodel_2/model_1/map_features/model/valence_embedding/embedding_lookup:output:0*
T0*'
_output_shapes
:���������|
:model_2/model_1/map_features/model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
5model_2/model_1/map_features/model/concatenate/concatConcatV2Fmodel_2/model_1/map_features/model/atom_sym_embedding/BiasAdd:output:0Hmodel_2/model_1/map_features/model/chiral_tag_embedding/BiasAdd:output:0Kmodel_2/model_1/map_features/model/hybridization_embedding/BiasAdd:output:0Vmodel_2/model_1/map_features/model/degree_embedding/embedding_lookup/Identity:output:0]model_2/model_1/map_features/model/formal_charge_embedding/embedding_lookup/Identity:output:0[model_2/model_1/map_features/model/is_aromatic_embedding/embedding_lookup/Identity:output:0[model_2/model_1/map_features/model/no_implicit_embedding/embedding_lookup/Identity:output:0Vmodel_2/model_1/map_features/model/num_Hs_embedding/embedding_lookup/Identity:output:0Wmodel_2/model_1/map_features/model/valence_embedding/embedding_lookup/Identity:output:0Cmodel_2/model_1/map_features/model/concatenate/concat/axis:output:0*
N	*
T0*'
_output_shapes
:����������
Gmodel_2/model_1/map_features/model/node_embedding/MatMul/ReadVariableOpReadVariableOpPmodel_2_model_1_map_features_model_node_embedding_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
8model_2/model_1/map_features/model/node_embedding/MatMulMatMul>model_2/model_1/map_features/model/concatenate/concat:output:0Omodel_2/model_1/map_features/model/node_embedding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Hmodel_2/model_1/map_features/model/node_embedding/BiasAdd/ReadVariableOpReadVariableOpQmodel_2_model_1_map_features_model_node_embedding_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
9model_2/model_1/map_features/model/node_embedding/BiasAddBiasAddBmodel_2/model_1/map_features/model/node_embedding/MatMul:product:0Pmodel_2/model_1/map_features/model/node_embedding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2model_2/model_1/map_features/model_1/reshape/ShapeShape,model_2/parse_example_2/Identity_14:output:0*
T0*
_output_shapes
::���
@model_2/model_1/map_features/model_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Bmodel_2/model_1/map_features/model_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Bmodel_2/model_1/map_features/model_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
:model_2/model_1/map_features/model_1/reshape/strided_sliceStridedSlice;model_2/model_1/map_features/model_1/reshape/Shape:output:0Imodel_2/model_1/map_features/model_1/reshape/strided_slice/stack:output:0Kmodel_2/model_1/map_features/model_1/reshape/strided_slice/stack_1:output:0Kmodel_2/model_1/map_features/model_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
<model_2/model_1/map_features/model_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
����������
:model_2/model_1/map_features/model_1/reshape/Reshape/shapePackCmodel_2/model_1/map_features/model_1/reshape/strided_slice:output:0Emodel_2/model_1/map_features/model_1/reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
4model_2/model_1/map_features/model_1/reshape/ReshapeReshape,model_2/parse_example_2/Identity_14:output:0Cmodel_2/model_1/map_features/model_1/reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:����������
Nmodel_2/model_1/map_features/model_1/bond_type_embedding/MatMul/ReadVariableOpReadVariableOpWmodel_2_model_1_map_features_model_1_bond_type_embedding_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
?model_2/model_1/map_features/model_1/bond_type_embedding/MatMulMatMul,model_2/parse_example_2/Identity_12:output:0Vmodel_2/model_1/map_features/model_1/bond_type_embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Omodel_2/model_1/map_features/model_1/bond_type_embedding/BiasAdd/ReadVariableOpReadVariableOpXmodel_2_model_1_map_features_model_1_bond_type_embedding_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
@model_2/model_1/map_features/model_1/bond_type_embedding/BiasAddBiasAddImodel_2/model_1/map_features/model_1/bond_type_embedding/MatMul:product:0Wmodel_2/model_1/map_features/model_1/bond_type_embedding/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Mmodel_2/model_1/map_features/model_1/is_conjugated_embedding/embedding_lookupResourceGatherTmodel_2_model_1_map_features_model_1_is_conjugated_embedding_embedding_lookup_315427,model_2/parse_example_2/Identity_16:output:0*
Tindices0	*g
_class]
[Yloc:@model_2/model_1/map_features/model_1/is_conjugated_embedding/embedding_lookup/315427*'
_output_shapes
:���������*
dtype0�
Vmodel_2/model_1/map_features/model_1/is_conjugated_embedding/embedding_lookup/IdentityIdentityVmodel_2/model_1/map_features/model_1/is_conjugated_embedding/embedding_lookup:output:0*
T0*'
_output_shapes
:����������
Kmodel_2/model_1/map_features/model_1/stereo_embedding/MatMul/ReadVariableOpReadVariableOpTmodel_2_model_1_map_features_model_1_stereo_embedding_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
<model_2/model_1/map_features/model_1/stereo_embedding/MatMulMatMul,model_2/parse_example_2/Identity_22:output:0Smodel_2/model_1/map_features/model_1/stereo_embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Lmodel_2/model_1/map_features/model_1/stereo_embedding/BiasAdd/ReadVariableOpReadVariableOpUmodel_2_model_1_map_features_model_1_stereo_embedding_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
=model_2/model_1/map_features/model_1/stereo_embedding/BiasAddBiasAddFmodel_2/model_1/map_features/model_1/stereo_embedding/MatMul:product:0Tmodel_2/model_1/map_features/model_1/stereo_embedding/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
>model_2/model_1/map_features/model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
9model_2/model_1/map_features/model_1/concatenate_1/concatConcatV2=model_2/model_1/map_features/model_1/reshape/Reshape:output:0Imodel_2/model_1/map_features/model_1/bond_type_embedding/BiasAdd:output:0_model_2/model_1/map_features/model_1/is_conjugated_embedding/embedding_lookup/Identity:output:0Fmodel_2/model_1/map_features/model_1/stereo_embedding/BiasAdd:output:0Gmodel_2/model_1/map_features/model_1/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
Imodel_2/model_1/map_features/model_1/edge_embedding/MatMul/ReadVariableOpReadVariableOpRmodel_2_model_1_map_features_model_1_edge_embedding_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
:model_2/model_1/map_features/model_1/edge_embedding/MatMulMatMulBmodel_2/model_1/map_features/model_1/concatenate_1/concat:output:0Qmodel_2/model_1/map_features/model_1/edge_embedding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Jmodel_2/model_1/map_features/model_1/edge_embedding/BiasAdd/ReadVariableOpReadVariableOpSmodel_2_model_1_map_features_model_1_edge_embedding_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;model_2/model_1/map_features/model_1/edge_embedding/BiasAddBiasAddDmodel_2/model_1/map_features/model_1/edge_embedding/MatMul:product:0Rmodel_2/model_1/map_features/model_1/edge_embedding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Bmodel_2/model_1/map_features/model_2/tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
>model_2/model_1/map_features/model_2/tf.expand_dims/ExpandDims
ExpandDims,model_2/parse_example_2/Identity_29:output:0Kmodel_2/model_1/map_features/model_2/tf.expand_dims/ExpandDims/dim:output:0*
T0*'
_output_shapes
:����������	
;model_2/model_1/map_features/model_2/tf.math.subtract/Sub/yConst*
_output_shapes	
:�*
dtype0*�
value�B��"�    ���=��L>���>���>   ?��?333?��L?fff?  �?�̌?���?ff�?33�?  �?���?���?ff�?33�?   @ff@��@33@��@   @ff&@��,@333@��9@  @@ffF@��L@33S@��Y@  `@fff@��l@33s@��y@  �@33�@ff�@���@�̌@  �@33�@ff�@���@�̜@  �@33�@ff�@���@�̬@  �@33�@ff�@���@�̼@  �@33�@ff�@���@���@  �@33�@ff�@���@���@  �@33�@ff�@���@���@  �@33�@ff�@���@���@   A��A33A��AffA  A��	A33A��AffA  A��A33A��AffA  A��A33A��AffA   A��!A33#A��$Aff&A  (A��)A33+A��,Aff.A  0A��1A333A��4Aff6A  8A��9A33;A��<Aff>A  @A��AA33CA��DAffFA  HA��IA33KA��LAffNA  PA��QA33SA��TAffVA  XA��YA33[A��\Aff^A  `A��aA33cA��dAfffA  hA��iA33kA��lAffnA  pA��qA33sA��tAffvA  xA��yA33{A��|Aff~A  �A�̀A���Aff�A33�A  �A�̄A���Aff�A33�A  �A�̈A���Aff�A33�A  �A�̌A���Aff�A33�A  �A�̐A���Aff�A33�A  �A�̔A���Aff�A33�A  �A�̘A���Aff�A33�A  �A�̜A���Aff�A33�A  �A�̠A���Aff�A33�A  �A�̤A���Aff�A33�A  �A�̨A���Aff�A33�A  �A�̬A���Aff�A33�A  �A�̰A���Aff�A33�A  �A�̴A���Aff�A33�A  �A�̸A���Aff�A33�A  �A�̼A���Aff�A33�A  �A���A���Aff�A33�A  �A���A���Aff�A33�A  �A���A���Aff�A33�A  �A�
9model_2/model_1/map_features/model_2/tf.math.subtract/SubSubGmodel_2/model_1/map_features/model_2/tf.expand_dims/ExpandDims:output:0Dmodel_2/model_1/map_features/model_2/tf.math.subtract/Sub/y:output:0*
T0*(
_output_shapes
:����������{
6model_2/model_1/map_features/model_2/tf.math.pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
4model_2/model_1/map_features/model_2/tf.math.pow/PowPow=model_2/model_1/map_features/model_2/tf.math.subtract/Sub:z:0?model_2/model_1/map_features/model_2/tf.math.pow/Pow/y:output:0*
T0*(
_output_shapes
:�����������
9model_2/model_1/map_features/model_2/tf.math.negative/NegNeg8model_2/model_1/map_features/model_2/tf.math.pow/Pow:z:0*
T0*(
_output_shapes
:�����������
>model_2/model_1/map_features/model_2/tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
<model_2/model_1/map_features/model_2/tf.math.truediv/truedivRealDiv=model_2/model_1/map_features/model_2/tf.math.negative/Neg:y:0Gmodel_2/model_1/map_features/model_2/tf.math.truediv/truediv/y:output:0*
T0*(
_output_shapes
:�����������
4model_2/model_1/map_features/model_2/tf.math.exp/ExpExp@model_2/model_1/map_features/model_2/tf.math.truediv/truediv:z:0*
T0*(
_output_shapes
:�����������
&model_2/model_1/map_features/ones_likeOnesLike@model_2/model/input.merge_batch_to_components/Reshape_8:output:0*
T0*#
_output_shapes
:���������|
:model_2/model_1/graph_update/edge_set_update/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
5model_2/model_1/graph_update/edge_set_update/GatherV2GatherV2Bmodel_2/model_1/map_features/model/node_embedding/BiasAdd:output:07model_2/model/input.merge_batch_to_components/Add_5:z:0Cmodel_2/model_1/graph_update/edge_set_update/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*(
_output_shapes
:����������~
<model_2/model_1/graph_update/edge_set_update/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
7model_2/model_1/graph_update/edge_set_update/GatherV2_1GatherV2Bmodel_2/model_1/map_features/model/node_embedding/BiasAdd:output:07model_2/model/input.merge_batch_to_components/Add_7:z:0Emodel_2/model_1/graph_update/edge_set_update/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*(
_output_shapes
:�����������
Lmodel_2/model_1/graph_update/edge_set_update/residual_next_state/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
Gmodel_2/model_1/graph_update/edge_set_update/residual_next_state/concatConcatV2Dmodel_2/model_1/map_features/model_1/edge_embedding/BiasAdd:output:0>model_2/model_1/graph_update/edge_set_update/GatherV2:output:0@model_2/model_1/graph_update/edge_set_update/GatherV2_1:output:0Umodel_2/model_1/graph_update/edge_set_update/residual_next_state/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
tmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/MatMul/ReadVariableOpReadVariableOp}model_2_model_1_graph_update_edge_set_update_residual_next_state_sequential_3_sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
emodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/MatMulMatMulPmodel_2/model_1/graph_update/edge_set_update/residual_next_state/concat:output:0|model_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
umodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp~model_2_model_1_graph_update_edge_set_update_residual_next_state_sequential_3_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
fmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/BiasAddBiasAddomodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/MatMul:product:0}model_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
xmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_edge_set_update_residual_next_state_sequential_3_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
imodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/MatMulMatMulomodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/BiasAdd:output:0�model_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
ymodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_edge_set_update_residual_next_state_sequential_3_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
jmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/BiasAddBiasAddsmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/MatMul:product:0�model_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/ReluRelusmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
xmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_edge_set_update_residual_next_state_sequential_3_sequential_2_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
imodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/MatMulMatMulumodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/Relu:activations:0�model_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
ymodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_edge_set_update_residual_next_state_sequential_3_sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
jmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/BiasAddBiasAddsmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/MatMul:product:0�model_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Dmodel_2/model_1/graph_update/edge_set_update/residual_next_state/AddAddV2smodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/BiasAdd:output:0Dmodel_2/model_1/map_features/model_1/edge_embedding/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
&model_2/model_1/graph_update/ones_likeOnesLike@model_2/model/input.merge_batch_to_components/Reshape_8:output:0*
T0*#
_output_shapes
:����������
7model_2/model_1/graph_update/node_set_update/pool/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
5model_2/model_1/graph_update/node_set_update/pool/SumSum@model_2/model/input.merge_batch_to_components/Reshape_6:output:0@model_2/model_1/graph_update/node_set_update/pool/Const:output:0*
T0*
_output_shapes
: �
7model_2/model_1/graph_update/node_set_update/pool/ShapeShapeHmodel_2/model_1/graph_update/edge_set_update/residual_next_state/Add:z:0*
T0*
_output_shapes
::���
Emodel_2/model_1/graph_update/node_set_update/pool/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Gmodel_2/model_1/graph_update/node_set_update/pool/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Gmodel_2/model_1/graph_update/node_set_update/pool/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
?model_2/model_1/graph_update/node_set_update/pool/strided_sliceStridedSlice@model_2/model_1/graph_update/node_set_update/pool/Shape:output:0Nmodel_2/model_1/graph_update/node_set_update/pool/strided_slice/stack:output:0Pmodel_2/model_1/graph_update/node_set_update/pool/strided_slice/stack_1:output:0Pmodel_2/model_1/graph_update/node_set_update/pool/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Dmodel_2/model_1/graph_update/node_set_update/pool/ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
>model_2/model_1/graph_update/node_set_update/pool/ones/ReshapeReshapeHmodel_2/model_1/graph_update/node_set_update/pool/strided_slice:output:0Mmodel_2/model_1/graph_update/node_set_update/pool/ones/Reshape/shape:output:0*
T0*
_output_shapes
:�
<model_2/model_1/graph_update/node_set_update/pool/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
6model_2/model_1/graph_update/node_set_update/pool/onesFillGmodel_2/model_1/graph_update/node_set_update/pool/ones/Reshape:output:0Emodel_2/model_1/graph_update/node_set_update/pool/ones/Const:output:0*
T0*#
_output_shapes
:����������
Dmodel_2/model_1/graph_update/node_set_update/pool/UnsortedSegmentSumUnsortedSegmentSum?model_2/model_1/graph_update/node_set_update/pool/ones:output:07model_2/model/input.merge_batch_to_components/Add_7:z:0>model_2/model_1/graph_update/node_set_update/pool/Sum:output:0*
Tindices0*
T0*#
_output_shapes
:����������
9model_2/model_1/graph_update/node_set_update/pool/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
7model_2/model_1/graph_update/node_set_update/pool/Sum_1Sum@model_2/model/input.merge_batch_to_components/Reshape_6:output:0Bmodel_2/model_1/graph_update/node_set_update/pool/Const_1:output:0*
T0*
_output_shapes
: �
Fmodel_2/model_1/graph_update/node_set_update/pool/UnsortedSegmentSum_1UnsortedSegmentSumHmodel_2/model_1/graph_update/edge_set_update/residual_next_state/Add:z:07model_2/model/input.merge_batch_to_components/Add_7:z:0@model_2/model_1/graph_update/node_set_update/pool/Sum_1:output:0*
Tindices0*
T0*(
_output_shapes
:�����������
@model_2/model_1/graph_update/node_set_update/pool/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
<model_2/model_1/graph_update/node_set_update/pool/ExpandDims
ExpandDimsMmodel_2/model_1/graph_update/node_set_update/pool/UnsortedSegmentSum:output:0Imodel_2/model_1/graph_update/node_set_update/pool/ExpandDims/dim:output:0*
T0*'
_output_shapes
:����������
<model_2/model_1/graph_update/node_set_update/pool/div_no_nanDivNoNanOmodel_2/model_1/graph_update/node_set_update/pool/UnsortedSegmentSum_1:output:0Emodel_2/model_1/graph_update/node_set_update/pool/ExpandDims:output:0*
T0*(
_output_shapes
:�����������
=model_2/model_1/graph_update/node_set_update/pool/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
8model_2/model_1/graph_update/node_set_update/pool/concatConcatV2@model_2/model_1/graph_update/node_set_update/pool/div_no_nan:z:0Omodel_2/model_1/graph_update/node_set_update/pool/UnsortedSegmentSum_1:output:0Fmodel_2/model_1/graph_update/node_set_update/pool/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
Nmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
Imodel_2/model_1/graph_update/node_set_update/residual_next_state_1/concatConcatV2Bmodel_2/model_1/map_features/model/node_embedding/BiasAdd:output:0Amodel_2/model_1/graph_update/node_set_update/pool/concat:output:0Wmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
zmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_4_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
kmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/MatMulMatMulRmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/concat:output:0�model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_4_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/BiasAddBiasAddumodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/MatMul:product:0�model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
imodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/ReluReluumodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
zmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_5_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
kmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/MatMulMatMulwmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/Relu:activations:0�model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_5_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/BiasAddBiasAddumodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/MatMul:product:0�model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
zmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_6_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
kmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/MatMulMatMulumodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/BiasAdd:output:0�model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_6_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/BiasAddBiasAddumodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/MatMul:product:0�model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
imodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/ReluReluumodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
zmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_7_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
kmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/MatMulMatMulwmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/Relu:activations:0�model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_node_set_update_residual_next_state_1_sequential_8_sequential_7_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/BiasAddBiasAddumodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/MatMul:product:0�model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
imodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/ReluReluumodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Fmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/AddAddV2wmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/Relu:activations:0Bmodel_2/model_1/map_features/model/node_embedding/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
(model_2/model_1/graph_update/ones_like_1OnesLike@model_2/model/input.merge_batch_to_components/Reshape_8:output:0*
T0*#
_output_shapes
:����������
>model_2/model_1/graph_update_1/edge_set_update_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9model_2/model_1/graph_update_1/edge_set_update_1/GatherV2GatherV2Jmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/Add:z:07model_2/model/input.merge_batch_to_components/Add_5:z:0Gmodel_2/model_1/graph_update_1/edge_set_update_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*(
_output_shapes
:�����������
@model_2/model_1/graph_update_1/edge_set_update_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;model_2/model_1/graph_update_1/edge_set_update_1/GatherV2_1GatherV2Jmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/Add:z:07model_2/model/input.merge_batch_to_components/Add_7:z:0Imodel_2/model_1/graph_update_1/edge_set_update_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*(
_output_shapes
:�����������
Rmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
Mmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/concatConcatV2Hmodel_2/model_1/graph_update/edge_set_update/residual_next_state/Add:z:0Bmodel_2/model_1/graph_update_1/edge_set_update_1/GatherV2:output:0Dmodel_2/model_1/graph_update_1/edge_set_update_1/GatherV2_1:output:0[model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_edge_set_update_1_residual_next_state_2_sequential_12_sequential_9_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
pmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/MatMulMatMulVmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/concat:output:0�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_edge_set_update_1_residual_next_state_2_sequential_12_sequential_9_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
qmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/BiasAddBiasAddzmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/MatMul:product:0�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_edge_set_update_1_residual_next_state_2_sequential_12_sequential_10_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
qmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/MatMulMatMulzmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/BiasAdd:output:0�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_edge_set_update_1_residual_next_state_2_sequential_12_sequential_10_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
rmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/BiasAddBiasAdd{model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/MatMul:product:0�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
omodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/ReluRelu{model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_edge_set_update_1_residual_next_state_2_sequential_12_sequential_11_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
qmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/MatMulMatMul}model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/Relu:activations:0�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_edge_set_update_1_residual_next_state_2_sequential_12_sequential_11_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
rmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/BiasAddBiasAdd{model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/MatMul:product:0�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Jmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/AddAddV2{model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/BiasAdd:output:0Hmodel_2/model_1/graph_update/edge_set_update/residual_next_state/Add:z:0*
T0*(
_output_shapes
:�����������
(model_2/model_1/graph_update_1/ones_likeOnesLike@model_2/model/input.merge_batch_to_components/Reshape_8:output:0*
T0*#
_output_shapes
:����������
=model_2/model_1/graph_update_1/node_set_update_1/pool_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
;model_2/model_1/graph_update_1/node_set_update_1/pool_1/SumSum@model_2/model/input.merge_batch_to_components/Reshape_6:output:0Fmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/Const:output:0*
T0*
_output_shapes
: �
=model_2/model_1/graph_update_1/node_set_update_1/pool_1/ShapeShapeNmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/Add:z:0*
T0*
_output_shapes
::���
Kmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Mmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Emodel_2/model_1/graph_update_1/node_set_update_1/pool_1/strided_sliceStridedSliceFmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/Shape:output:0Tmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/strided_slice/stack:output:0Vmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/strided_slice/stack_1:output:0Vmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Jmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
Dmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/ones/ReshapeReshapeNmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/strided_slice:output:0Smodel_2/model_1/graph_update_1/node_set_update_1/pool_1/ones/Reshape/shape:output:0*
T0*
_output_shapes
:�
Bmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
<model_2/model_1/graph_update_1/node_set_update_1/pool_1/onesFillMmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/ones/Reshape:output:0Kmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/ones/Const:output:0*
T0*#
_output_shapes
:����������
Jmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/UnsortedSegmentSumUnsortedSegmentSumEmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/ones:output:07model_2/model/input.merge_batch_to_components/Add_7:z:0Dmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/Sum:output:0*
Tindices0*
T0*#
_output_shapes
:����������
?model_2/model_1/graph_update_1/node_set_update_1/pool_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
=model_2/model_1/graph_update_1/node_set_update_1/pool_1/Sum_1Sum@model_2/model/input.merge_batch_to_components/Reshape_6:output:0Hmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/Const_1:output:0*
T0*
_output_shapes
: �
Lmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/UnsortedSegmentSum_1UnsortedSegmentSumNmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/Add:z:07model_2/model/input.merge_batch_to_components/Add_7:z:0Fmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/Sum_1:output:0*
Tindices0*
T0*(
_output_shapes
:�����������
Fmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Bmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/ExpandDims
ExpandDimsSmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/UnsortedSegmentSum:output:0Omodel_2/model_1/graph_update_1/node_set_update_1/pool_1/ExpandDims/dim:output:0*
T0*'
_output_shapes
:����������
Bmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/div_no_nanDivNoNanUmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/UnsortedSegmentSum_1:output:0Kmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/ExpandDims:output:0*
T0*(
_output_shapes
:�����������
Cmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
>model_2/model_1/graph_update_1/node_set_update_1/pool_1/concatConcatV2Fmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/div_no_nan:z:0Umodel_2/model_1/graph_update_1/node_set_update_1/pool_1/UnsortedSegmentSum_1:output:0Lmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
Rmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
Mmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/concatConcatV2Jmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/Add:z:0Gmodel_2/model_1/graph_update_1/node_set_update_1/pool_1/concat:output:0[model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_13_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
rmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/MatMulMatMulVmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/concat:output:0�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_13_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
smodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/BiasAddBiasAdd|model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/MatMul:product:0�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
pmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/ReluRelu|model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_14_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
rmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/MatMulMatMul~model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/Relu:activations:0�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_14_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
smodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/BiasAddBiasAdd|model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/MatMul:product:0�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_15_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
rmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/MatMulMatMul|model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/BiasAdd:output:0�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_15_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
smodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/BiasAddBiasAdd|model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/MatMul:product:0�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
pmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/ReluRelu|model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_16_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
rmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/MatMulMatMul~model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/Relu:activations:0�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_1_node_set_update_1_residual_next_state_3_sequential_17_sequential_16_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
smodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/BiasAddBiasAdd|model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/MatMul:product:0�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
pmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/ReluRelu|model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Jmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/AddAddV2~model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/Relu:activations:0Jmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/Add:z:0*
T0*(
_output_shapes
:�����������
*model_2/model_1/graph_update_1/ones_like_1OnesLike@model_2/model/input.merge_batch_to_components/Reshape_8:output:0*
T0*#
_output_shapes
:����������
>model_2/model_1/graph_update_2/edge_set_update_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9model_2/model_1/graph_update_2/edge_set_update_2/GatherV2GatherV2Nmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/Add:z:07model_2/model/input.merge_batch_to_components/Add_5:z:0Gmodel_2/model_1/graph_update_2/edge_set_update_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*(
_output_shapes
:�����������
@model_2/model_1/graph_update_2/edge_set_update_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;model_2/model_1/graph_update_2/edge_set_update_2/GatherV2_1GatherV2Nmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/Add:z:07model_2/model/input.merge_batch_to_components/Add_7:z:0Imodel_2/model_1/graph_update_2/edge_set_update_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*(
_output_shapes
:�����������
Rmodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
Mmodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/concatConcatV2Nmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/Add:z:0Bmodel_2/model_1/graph_update_2/edge_set_update_2/GatherV2:output:0Dmodel_2/model_1/graph_update_2/edge_set_update_2/GatherV2_1:output:0[model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_edge_set_update_2_residual_next_state_4_sequential_21_sequential_18_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
rmodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/MatMulMatMulVmodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/concat:output:0�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_edge_set_update_2_residual_next_state_4_sequential_21_sequential_18_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
smodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/BiasAddBiasAdd|model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/MatMul:product:0�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_edge_set_update_2_residual_next_state_4_sequential_21_sequential_19_dense_15_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
rmodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/MatMulMatMul|model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/BiasAdd:output:0�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_edge_set_update_2_residual_next_state_4_sequential_21_sequential_19_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
smodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/BiasAddBiasAdd|model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/MatMul:product:0�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
pmodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/ReluRelu|model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_edge_set_update_2_residual_next_state_4_sequential_21_sequential_20_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
rmodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/MatMulMatMul~model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/Relu:activations:0�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_edge_set_update_2_residual_next_state_4_sequential_21_sequential_20_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
smodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/BiasAddBiasAdd|model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/MatMul:product:0�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Jmodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/AddAddV2|model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/BiasAdd:output:0Nmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/Add:z:0*
T0*(
_output_shapes
:�����������
(model_2/model_1/graph_update_2/ones_likeOnesLike@model_2/model/input.merge_batch_to_components/Reshape_8:output:0*
T0*#
_output_shapes
:����������
=model_2/model_1/graph_update_2/node_set_update_2/pool_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
;model_2/model_1/graph_update_2/node_set_update_2/pool_2/SumSum@model_2/model/input.merge_batch_to_components/Reshape_6:output:0Fmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/Const:output:0*
T0*
_output_shapes
: �
=model_2/model_1/graph_update_2/node_set_update_2/pool_2/ShapeShapeNmodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/Add:z:0*
T0*
_output_shapes
::���
Kmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Mmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Emodel_2/model_1/graph_update_2/node_set_update_2/pool_2/strided_sliceStridedSliceFmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/Shape:output:0Tmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/strided_slice/stack:output:0Vmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/strided_slice/stack_1:output:0Vmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Jmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
Dmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/ones/ReshapeReshapeNmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/strided_slice:output:0Smodel_2/model_1/graph_update_2/node_set_update_2/pool_2/ones/Reshape/shape:output:0*
T0*
_output_shapes
:�
Bmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
<model_2/model_1/graph_update_2/node_set_update_2/pool_2/onesFillMmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/ones/Reshape:output:0Kmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/ones/Const:output:0*
T0*#
_output_shapes
:����������
Jmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/UnsortedSegmentSumUnsortedSegmentSumEmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/ones:output:07model_2/model/input.merge_batch_to_components/Add_7:z:0Dmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/Sum:output:0*
Tindices0*
T0*#
_output_shapes
:����������
?model_2/model_1/graph_update_2/node_set_update_2/pool_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
=model_2/model_1/graph_update_2/node_set_update_2/pool_2/Sum_1Sum@model_2/model/input.merge_batch_to_components/Reshape_6:output:0Hmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/Const_1:output:0*
T0*
_output_shapes
: �
Lmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/UnsortedSegmentSum_1UnsortedSegmentSumNmodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/Add:z:07model_2/model/input.merge_batch_to_components/Add_7:z:0Fmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/Sum_1:output:0*
Tindices0*
T0*(
_output_shapes
:�����������
Fmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Bmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/ExpandDims
ExpandDimsSmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/UnsortedSegmentSum:output:0Omodel_2/model_1/graph_update_2/node_set_update_2/pool_2/ExpandDims/dim:output:0*
T0*'
_output_shapes
:����������
Bmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/div_no_nanDivNoNanUmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/UnsortedSegmentSum_1:output:0Kmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/ExpandDims:output:0*
T0*(
_output_shapes
:�����������
Cmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
>model_2/model_1/graph_update_2/node_set_update_2/pool_2/concatConcatV2Fmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/div_no_nan:z:0Umodel_2/model_1/graph_update_2/node_set_update_2/pool_2/UnsortedSegmentSum_1:output:0Lmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
Rmodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
Mmodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/concatConcatV2Nmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/Add:z:0Gmodel_2/model_1/graph_update_2/node_set_update_2/pool_2/concat:output:0[model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_22_dense_17_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
rmodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/MatMulMatMulVmodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/concat:output:0�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_22_dense_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
smodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/BiasAddBiasAdd|model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/MatMul:product:0�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
pmodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/ReluRelu|model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_23_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
rmodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/MatMulMatMul~model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/Relu:activations:0�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_23_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
smodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/BiasAddBiasAdd|model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/MatMul:product:0�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_24_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
rmodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/MatMulMatMul|model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/BiasAdd:output:0�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_24_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
smodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/BiasAddBiasAdd|model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/MatMul:product:0�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
pmodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/ReluRelu|model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/MatMul/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_25_dense_20_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
rmodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/MatMulMatMul~model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/Relu:activations:0�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/BiasAdd/ReadVariableOpReadVariableOp�model_2_model_1_graph_update_2_node_set_update_2_residual_next_state_5_sequential_26_sequential_25_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
smodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/BiasAddBiasAdd|model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/MatMul:product:0�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
pmodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/ReluRelu|model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Jmodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/AddAddV2~model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/Relu:activations:0Nmodel_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/Add:z:0*
T0*(
_output_shapes
:�����������
*model_2/model_1/graph_update_2/ones_like_1OnesLike@model_2/model/input.merge_batch_to_components/Reshape_8:output:0*
T0*#
_output_shapes
:���������r
(model_2/model_1/structured_readout/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
&model_2/model_1/structured_readout/SumSum@model_2/model/input.merge_batch_to_components/Reshape_5:output:01model_2/model_1/structured_readout/Const:output:0*
T0*
_output_shapes
: �
6model_2/model_1/structured_readout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
8model_2/model_1/structured_readout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
8model_2/model_1/structured_readout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0model_2/model_1/structured_readout/strided_sliceStridedSlice7model_2/model/input.merge_batch_to_components/Add_3:z:0?model_2/model_1/structured_readout/strided_slice/stack:output:0Amodel_2/model_1/structured_readout/strided_slice/stack_1:output:0Amodel_2/model_1/structured_readout/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask�
8model_2/model_1/structured_readout/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
:model_2/model_1/structured_readout/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
:model_2/model_1/structured_readout/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
2model_2/model_1/structured_readout/strided_slice_1StridedSlice7model_2/model/input.merge_batch_to_components/Add_3:z:0Amodel_2/model_1/structured_readout/strided_slice_1/stack:output:0Cmodel_2/model_1/structured_readout/strided_slice_1/stack_1:output:0Cmodel_2/model_1/structured_readout/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*
end_mask�
3model_2/model_1/structured_readout/assert_less/LessLess9model_2/model_1/structured_readout/strided_slice:output:0;model_2/model_1/structured_readout/strided_slice_1:output:0*
T0*#
_output_shapes
:���������~
4model_2/model_1/structured_readout/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
2model_2/model_1/structured_readout/assert_less/AllAll7model_2/model_1/structured_readout/assert_less/Less:z:0=model_2/model_1/structured_readout/assert_less/Const:output:0*
_output_shapes
: �
;model_2/model_1/structured_readout/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*@
value7B5 B/Not strictly sorted by target: '_readout/shape'�
=model_2/model_1/structured_readout/assert_less/Assert/Const_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Condition x < y did not hold element-wise:�
=model_2/model_1/structured_readout/assert_less/Assert/Const_2Const*
_output_shapes
: *
dtype0*J
valueAB? B9x (model_2/model_1/structured_readout/strided_slice:0) = �
=model_2/model_1/structured_readout/assert_less/Assert/Const_3Const*
_output_shapes
: *
dtype0*L
valueCBA B;y (model_2/model_1/structured_readout/strided_slice_1:0) = �
Cmodel_2/model_1/structured_readout/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*@
value7B5 B/Not strictly sorted by target: '_readout/shape'�
Cmodel_2/model_1/structured_readout/assert_less/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Condition x < y did not hold element-wise:�
Cmodel_2/model_1/structured_readout/assert_less/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*J
valueAB? B9x (model_2/model_1/structured_readout/strided_slice:0) = �
Cmodel_2/model_1/structured_readout/assert_less/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*L
valueCBA B;y (model_2/model_1/structured_readout/strided_slice_1:0) = �
<model_2/model_1/structured_readout/assert_less/Assert/AssertAssert;model_2/model_1/structured_readout/assert_less/All:output:0Lmodel_2/model_1/structured_readout/assert_less/Assert/Assert/data_0:output:0Lmodel_2/model_1/structured_readout/assert_less/Assert/Assert/data_1:output:0Lmodel_2/model_1/structured_readout/assert_less/Assert/Assert/data_2:output:09model_2/model_1/structured_readout/strided_slice:output:0Lmodel_2/model_1/structured_readout/assert_less/Assert/Assert/data_4:output:0;model_2/model_1/structured_readout/strided_slice_1:output:06^model_2/parse_example_2/assert_shapes/Assert_3/Assert*
T

2*
_output_shapes
 v
4model_2/model_1/structured_readout/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : �
0model_2/model_1/structured_readout/concat/concatIdentity7model_2/model/input.merge_batch_to_components/Add_3:z:0*
T0*#
_output_shapes
:���������w
,model_2/model_1/structured_readout/sort/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
+model_2/model_1/structured_readout/sort/NegNeg9model_2/model_1/structured_readout/concat/concat:output:0*
T0*#
_output_shapes
:���������o
-model_2/model_1/structured_readout/sort/sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
+model_2/model_1/structured_readout/sort/subSub/model_2/model_1/structured_readout/sort/Neg:y:06model_2/model_1/structured_readout/sort/sub/y:output:0*
T0*#
_output_shapes
:����������
-model_2/model_1/structured_readout/sort/ShapeShape/model_2/model_1/structured_readout/sort/sub:z:0*
T0*
_output_shapes
::���
;model_2/model_1/structured_readout/sort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
=model_2/model_1/structured_readout/sort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
=model_2/model_1/structured_readout/sort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5model_2/model_1/structured_readout/sort/strided_sliceStridedSlice6model_2/model_1/structured_readout/sort/Shape:output:0Dmodel_2/model_1/structured_readout/sort/strided_slice/stack:output:0Fmodel_2/model_1/structured_readout/sort/strided_slice/stack_1:output:0Fmodel_2/model_1/structured_readout/sort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,model_2/model_1/structured_readout/sort/RankConst*
_output_shapes
: *
dtype0*
value	B :�
.model_2/model_1/structured_readout/sort/TopKV2TopKV2/model_2/model_1/structured_readout/sort/sub:z:0>model_2/model_1/structured_readout/sort/strided_slice:output:0*
T0*2
_output_shapes 
:���������:����������
-model_2/model_1/structured_readout/sort/Neg_1Neg7model_2/model_1/structured_readout/sort/TopKV2:values:0*
T0*#
_output_shapes
:���������q
/model_2/model_1/structured_readout/sort/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
-model_2/model_1/structured_readout/sort/sub_1Sub1model_2/model_1/structured_readout/sort/Neg_1:y:08model_2/model_1/structured_readout/sort/sub_1/y:output:0*
T0*#
_output_shapes
:���������p
.model_2/model_1/structured_readout/range/startConst*
_output_shapes
: *
dtype0*
value	B : p
.model_2/model_1/structured_readout/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
(model_2/model_1/structured_readout/rangeRange7model_2/model_1/structured_readout/range/start:output:0/model_2/model_1/structured_readout/Sum:output:07model_2/model_1/structured_readout/range/delta:output:0*#
_output_shapes
:����������
7model_2/model_1/structured_readout/assert_equal_1/EqualEqual1model_2/model_1/structured_readout/sort/sub_1:z:01model_2/model_1/structured_readout/range:output:0*
T0*#
_output_shapes
:����������
7model_2/model_1/structured_readout/assert_equal_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
5model_2/model_1/structured_readout/assert_equal_1/AllAll;model_2/model_1/structured_readout/assert_equal_1/Equal:z:0@model_2/model_1/structured_readout/assert_equal_1/Const:output:0*
_output_shapes
: �
>model_2/model_1/structured_readout/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*@
value7B5 B/Target indices not equal to range(readout_size)�
@model_2/model_1/structured_readout/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
@model_2/model_1/structured_readout/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*G
value>B< B6x (model_2/model_1/structured_readout/sort/sub_1:0) = �
@model_2/model_1/structured_readout/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*B
value9B7 B1y (model_2/model_1/structured_readout/range:0) = �
Fmodel_2/model_1/structured_readout/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*@
value7B5 B/Target indices not equal to range(readout_size)�
Fmodel_2/model_1/structured_readout/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:�
Fmodel_2/model_1/structured_readout/assert_equal_1/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*G
value>B< B6x (model_2/model_1/structured_readout/sort/sub_1:0) = �
Fmodel_2/model_1/structured_readout/assert_equal_1/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*B
value9B7 B1y (model_2/model_1/structured_readout/range:0) = �
?model_2/model_1/structured_readout/assert_equal_1/Assert/AssertAssert>model_2/model_1/structured_readout/assert_equal_1/All:output:0Omodel_2/model_1/structured_readout/assert_equal_1/Assert/Assert/data_0:output:0Omodel_2/model_1/structured_readout/assert_equal_1/Assert/Assert/data_1:output:0Omodel_2/model_1/structured_readout/assert_equal_1/Assert/Assert/data_2:output:01model_2/model_1/structured_readout/sort/sub_1:z:0Omodel_2/model_1/structured_readout/assert_equal_1/Assert/Assert/data_4:output:01model_2/model_1/structured_readout/range:output:0=^model_2/model_1/structured_readout/assert_less/Assert/Assert*
T

2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
	summarized�
+model_2/model_1/structured_readout/IdentityIdentity@model_2/model/input.merge_batch_to_components/Reshape_2:output:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0	*#
_output_shapes
:����������
-model_2/model_1/structured_readout/Identity_1Identity@model_2/model/input.merge_batch_to_components/Reshape_3:output:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*#
_output_shapes
:����������
-model_2/model_1/structured_readout/Identity_2Identity.model_2/model_1/graph_update_2/ones_like_1:y:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*#
_output_shapes
:����������
-model_2/model_1/structured_readout/Identity_3Identity7model_2/model/input.merge_batch_to_components/Add_1:z:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*#
_output_shapes
:����������
-model_2/model_1/structured_readout/Identity_4Identity7model_2/model/input.merge_batch_to_components/Add_3:z:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*#
_output_shapes
:����������
-model_2/model_1/structured_readout/Identity_5Identity@model_2/model/input.merge_batch_to_components/Reshape_8:output:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*#
_output_shapes
:����������
-model_2/model_1/structured_readout/Identity_6Identity7model_2/model/input.merge_batch_to_components/Add_5:z:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*#
_output_shapes
:����������
-model_2/model_1/structured_readout/Identity_7Identity7model_2/model/input.merge_batch_to_components/Add_7:z:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*#
_output_shapes
:����������
-model_2/model_1/structured_readout/Identity_8IdentityNmodel_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/Add:z:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*(
_output_shapes
:�����������
-model_2/model_1/structured_readout/Identity_9IdentityAmodel_2/model/input.merge_batch_to_components/Reshape_10:output:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*#
_output_shapes
:����������
.model_2/model_1/structured_readout/Identity_10Identity7model_2/model/input.merge_batch_to_components/Add_9:z:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*#
_output_shapes
:����������
.model_2/model_1/structured_readout/Identity_11Identity8model_2/model/input.merge_batch_to_components/Add_11:z:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*#
_output_shapes
:����������
.model_2/model_1/structured_readout/Identity_12Identity8model_2/model_1/map_features/model_2/tf.math.exp/Exp:y:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*(
_output_shapes
:�����������
.model_2/model_1/structured_readout/Identity_13IdentityAmodel_2/model/input.merge_batch_to_components/Reshape_12:output:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*#
_output_shapes
:����������
.model_2/model_1/structured_readout/Identity_14Identity@model_2/model/input.merge_batch_to_components/Reshape_5:output:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*#
_output_shapes
:����������
.model_2/model_1/structured_readout/Identity_15IdentityNmodel_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/Add:z:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*(
_output_shapes
:�����������
.model_2/model_1/structured_readout/Identity_16Identity@model_2/model/input.merge_batch_to_components/Reshape_6:output:0@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert*
T0*#
_output_shapes
:���������r
0model_2/model_1/structured_readout/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+model_2/model_1/structured_readout/GatherV2GatherV27model_2/model_1/structured_readout/Identity_15:output:06model_2/model_1/structured_readout/Identity_3:output:09model_2/model_1/structured_readout/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*(
_output_shapes
:�����������
Jmodel_2/model_1/sequential_29/sequential_27/dense_21/MatMul/ReadVariableOpReadVariableOpSmodel_2_model_1_sequential_29_sequential_27_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
;model_2/model_1/sequential_29/sequential_27/dense_21/MatMulMatMul4model_2/model_1/structured_readout/GatherV2:output:0Rmodel_2/model_1/sequential_29/sequential_27/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Kmodel_2/model_1/sequential_29/sequential_27/dense_21/BiasAdd/ReadVariableOpReadVariableOpTmodel_2_model_1_sequential_29_sequential_27_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model_2/model_1/sequential_29/sequential_27/dense_21/BiasAddBiasAddEmodel_2/model_1/sequential_29/sequential_27/dense_21/MatMul:product:0Smodel_2/model_1/sequential_29/sequential_27/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
9model_2/model_1/sequential_29/sequential_27/dense_21/ReluReluEmodel_2/model_1/sequential_29/sequential_27/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
Jmodel_2/model_1/sequential_29/sequential_28/dense_22/MatMul/ReadVariableOpReadVariableOpSmodel_2_model_1_sequential_29_sequential_28_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
;model_2/model_1/sequential_29/sequential_28/dense_22/MatMulMatMulGmodel_2/model_1/sequential_29/sequential_27/dense_21/Relu:activations:0Rmodel_2/model_1/sequential_29/sequential_28/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Kmodel_2/model_1/sequential_29/sequential_28/dense_22/BiasAdd/ReadVariableOpReadVariableOpTmodel_2_model_1_sequential_29_sequential_28_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model_2/model_1/sequential_29/sequential_28/dense_22/BiasAddBiasAddEmodel_2/model_1/sequential_29/sequential_28/dense_22/MatMul:product:0Smodel_2/model_1/sequential_29/sequential_28/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
9model_2/model_1/sequential_29/sequential_28/dense_22/ReluReluEmodel_2/model_1/sequential_29/sequential_28/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:����������n
,model_2/model_1/repeat_vector/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
(model_2/model_1/repeat_vector/ExpandDims
ExpandDimsGmodel_2/model_1/sequential_29/sequential_28/dense_22/Relu:activations:05model_2/model_1/repeat_vector/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������x
#model_2/model_1/repeat_vector/stackConst*
_output_shapes
:*
dtype0*!
valueB"         �
"model_2/model_1/repeat_vector/TileTile1model_2/model_1/repeat_vector/ExpandDims:output:0,model_2/model_1/repeat_vector/stack:output:0*
T0*,
_output_shapes
:�����������
model_2/model_1/gru/ShapeShape+model_2/model_1/repeat_vector/Tile:output:0*
T0*
_output_shapes
::��q
'model_2/model_1/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)model_2/model_1/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)model_2/model_1/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!model_2/model_1/gru/strided_sliceStridedSlice"model_2/model_1/gru/Shape:output:00model_2/model_1/gru/strided_slice/stack:output:02model_2/model_1/gru/strided_slice/stack_1:output:02model_2/model_1/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_2/model_1/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
 model_2/model_1/gru/zeros/packedPack*model_2/model_1/gru/strided_slice:output:0+model_2/model_1/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
model_2/model_1/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_2/model_1/gru/zerosFill)model_2/model_1/gru/zeros/packed:output:0(model_2/model_1/gru/zeros/Const:output:0*
T0*'
_output_shapes
:����������
'model_2/model_1/gru/Read/ReadVariableOpReadVariableOp0model_2_model_1_gru_read_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model_2/model_1/gru/IdentityIdentity/model_2/model_1/gru/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
)model_2/model_1/gru/Read_1/ReadVariableOpReadVariableOp2model_2_model_1_gru_read_1_readvariableop_resource*
_output_shapes

:*
dtype0�
model_2/model_1/gru/Identity_1Identity1model_2/model_1/gru/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

:�
)model_2/model_1/gru/Read_2/ReadVariableOpReadVariableOp2model_2_model_1_gru_read_2_readvariableop_resource*
_output_shapes

:*
dtype0�
model_2/model_1/gru/Identity_2Identity1model_2/model_1/gru/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:�
#model_2/model_1/gru/PartitionedCallPartitionedCall+model_2/model_1/repeat_vector/Tile:output:0"model_2/model_1/gru/zeros:output:0%model_2/model_1/gru/Identity:output:0'model_2/model_1/gru/Identity_1:output:0'model_2/model_1/gru/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:���������:���������:���������: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference_standard_gru_315943�
model_2/model_1/softmax/SoftmaxSoftmax,model_2/model_1/gru/PartitionedCall:output:1*
T0*+
_output_shapes
:���������|
IdentityIdentity)model_2/model_1/softmax/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:����������N
NoOpNoOpv^model_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/BiasAdd/ReadVariableOpu^model_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/MatMul/ReadVariableOpz^model_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/BiasAdd/ReadVariableOpy^model_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/MatMul/ReadVariableOpz^model_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/BiasAdd/ReadVariableOpy^model_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/MatMul/ReadVariableOp|^model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/BiasAdd/ReadVariableOp{^model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/MatMul/ReadVariableOp|^model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/BiasAdd/ReadVariableOp{^model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/MatMul/ReadVariableOp|^model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/BiasAdd/ReadVariableOp{^model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/MatMul/ReadVariableOp|^model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/BiasAdd/ReadVariableOp{^model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/MatMul/ReadVariableOp�^model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/MatMul/ReadVariableOp�^model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/MatMul/ReadVariableOp�^model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/MatMul/ReadVariableOp�^model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/MatMul/ReadVariableOp�^model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/MatMul/ReadVariableOp�^model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/MatMul/ReadVariableOp�^model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/MatMul/ReadVariableOp�^model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/MatMul/ReadVariableOp�^model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/MatMul/ReadVariableOp�^model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/MatMul/ReadVariableOp�^model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/MatMul/ReadVariableOp�^model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/MatMul/ReadVariableOp�^model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/MatMul/ReadVariableOp�^model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/BiasAdd/ReadVariableOp�^model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/MatMul/ReadVariableOp(^model_2/model_1/gru/Read/ReadVariableOp*^model_2/model_1/gru/Read_1/ReadVariableOp*^model_2/model_1/gru/Read_2/ReadVariableOpM^model_2/model_1/map_features/model/atom_sym_embedding/BiasAdd/ReadVariableOpL^model_2/model_1/map_features/model/atom_sym_embedding/MatMul/ReadVariableOpO^model_2/model_1/map_features/model/chiral_tag_embedding/BiasAdd/ReadVariableOpN^model_2/model_1/map_features/model/chiral_tag_embedding/MatMul/ReadVariableOpE^model_2/model_1/map_features/model/degree_embedding/embedding_lookupL^model_2/model_1/map_features/model/formal_charge_embedding/embedding_lookupR^model_2/model_1/map_features/model/hybridization_embedding/BiasAdd/ReadVariableOpQ^model_2/model_1/map_features/model/hybridization_embedding/MatMul/ReadVariableOpJ^model_2/model_1/map_features/model/is_aromatic_embedding/embedding_lookupJ^model_2/model_1/map_features/model/no_implicit_embedding/embedding_lookupI^model_2/model_1/map_features/model/node_embedding/BiasAdd/ReadVariableOpH^model_2/model_1/map_features/model/node_embedding/MatMul/ReadVariableOpE^model_2/model_1/map_features/model/num_Hs_embedding/embedding_lookupF^model_2/model_1/map_features/model/valence_embedding/embedding_lookupP^model_2/model_1/map_features/model_1/bond_type_embedding/BiasAdd/ReadVariableOpO^model_2/model_1/map_features/model_1/bond_type_embedding/MatMul/ReadVariableOpK^model_2/model_1/map_features/model_1/edge_embedding/BiasAdd/ReadVariableOpJ^model_2/model_1/map_features/model_1/edge_embedding/MatMul/ReadVariableOpN^model_2/model_1/map_features/model_1/is_conjugated_embedding/embedding_lookupM^model_2/model_1/map_features/model_1/stereo_embedding/BiasAdd/ReadVariableOpL^model_2/model_1/map_features/model_1/stereo_embedding/MatMul/ReadVariableOpL^model_2/model_1/sequential_29/sequential_27/dense_21/BiasAdd/ReadVariableOpK^model_2/model_1/sequential_29/sequential_27/dense_21/MatMul/ReadVariableOpL^model_2/model_1/sequential_29/sequential_28/dense_22/BiasAdd/ReadVariableOpK^model_2/model_1/sequential_29/sequential_28/dense_22/MatMul/ReadVariableOp@^model_2/model_1/structured_readout/assert_equal_1/Assert/Assert=^model_2/model_1/structured_readout/assert_less/Assert/Assertc^model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assertz^model_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertI^model_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Asserte^model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert|^model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertK^model_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Asserte^model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert|^model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertK^model_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Asserte^model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert|^model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertK^model_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Asserte^model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert|^model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertK^model_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Asserte^model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert|^model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertK^model_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Asserte^model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert|^model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertK^model_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Assert4^model_2/parse_example_2/assert_shapes/Assert/Assert6^model_2/parse_example_2/assert_shapes/Assert_1/Assert6^model_2/parse_example_2/assert_shapes/Assert_2/Assert6^model_2/parse_example_2/assert_shapes/Assert_3/Assert*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2�
umodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/BiasAdd/ReadVariableOpumodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/BiasAdd/ReadVariableOp2�
tmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/MatMul/ReadVariableOptmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential/dense/MatMul/ReadVariableOp2�
ymodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/BiasAdd/ReadVariableOpymodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/BiasAdd/ReadVariableOp2�
xmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/MatMul/ReadVariableOpxmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_1/dense_1/MatMul/ReadVariableOp2�
ymodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/BiasAdd/ReadVariableOpymodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/BiasAdd/ReadVariableOp2�
xmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/MatMul/ReadVariableOpxmodel_2/model_1/graph_update/edge_set_update/residual_next_state/sequential_3/sequential_2/dense_2/MatMul/ReadVariableOp2�
{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/BiasAdd/ReadVariableOp{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/BiasAdd/ReadVariableOp2�
zmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/MatMul/ReadVariableOpzmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_4/dense_3/MatMul/ReadVariableOp2�
{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/BiasAdd/ReadVariableOp{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/BiasAdd/ReadVariableOp2�
zmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/MatMul/ReadVariableOpzmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_5/dense_4/MatMul/ReadVariableOp2�
{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/BiasAdd/ReadVariableOp{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/BiasAdd/ReadVariableOp2�
zmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/MatMul/ReadVariableOpzmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_6/dense_5/MatMul/ReadVariableOp2�
{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/BiasAdd/ReadVariableOp{model_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/BiasAdd/ReadVariableOp2�
zmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/MatMul/ReadVariableOpzmodel_2/model_1/graph_update/node_set_update/residual_next_state_1/sequential_8/sequential_7/dense_6/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/BiasAdd/ReadVariableOp2�
�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/MatMul/ReadVariableOp�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_10/dense_8/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/BiasAdd/ReadVariableOp2�
�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/MatMul/ReadVariableOp�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_11/dense_9/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/BiasAdd/ReadVariableOp2�
model_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/MatMul/ReadVariableOpmodel_2/model_1/graph_update_1/edge_set_update_1/residual_next_state_2/sequential_12/sequential_9/dense_7/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/BiasAdd/ReadVariableOp2�
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/MatMul/ReadVariableOp�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_13/dense_10/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/BiasAdd/ReadVariableOp2�
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/MatMul/ReadVariableOp�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_14/dense_11/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/BiasAdd/ReadVariableOp2�
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/MatMul/ReadVariableOp�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_15/dense_12/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/BiasAdd/ReadVariableOp2�
�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/MatMul/ReadVariableOp�model_2/model_1/graph_update_1/node_set_update_1/residual_next_state_3/sequential_17/sequential_16/dense_13/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/BiasAdd/ReadVariableOp2�
�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/MatMul/ReadVariableOp�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_18/dense_14/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/BiasAdd/ReadVariableOp2�
�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/MatMul/ReadVariableOp�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_19/dense_15/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/BiasAdd/ReadVariableOp2�
�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/MatMul/ReadVariableOp�model_2/model_1/graph_update_2/edge_set_update_2/residual_next_state_4/sequential_21/sequential_20/dense_16/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/BiasAdd/ReadVariableOp2�
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/MatMul/ReadVariableOp�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_22/dense_17/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/BiasAdd/ReadVariableOp2�
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/MatMul/ReadVariableOp�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_23/dense_18/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/BiasAdd/ReadVariableOp2�
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/MatMul/ReadVariableOp�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_24/dense_19/MatMul/ReadVariableOp2�
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/BiasAdd/ReadVariableOp�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/BiasAdd/ReadVariableOp2�
�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/MatMul/ReadVariableOp�model_2/model_1/graph_update_2/node_set_update_2/residual_next_state_5/sequential_26/sequential_25/dense_20/MatMul/ReadVariableOp2R
'model_2/model_1/gru/Read/ReadVariableOp'model_2/model_1/gru/Read/ReadVariableOp2V
)model_2/model_1/gru/Read_1/ReadVariableOp)model_2/model_1/gru/Read_1/ReadVariableOp2V
)model_2/model_1/gru/Read_2/ReadVariableOp)model_2/model_1/gru/Read_2/ReadVariableOp2�
Lmodel_2/model_1/map_features/model/atom_sym_embedding/BiasAdd/ReadVariableOpLmodel_2/model_1/map_features/model/atom_sym_embedding/BiasAdd/ReadVariableOp2�
Kmodel_2/model_1/map_features/model/atom_sym_embedding/MatMul/ReadVariableOpKmodel_2/model_1/map_features/model/atom_sym_embedding/MatMul/ReadVariableOp2�
Nmodel_2/model_1/map_features/model/chiral_tag_embedding/BiasAdd/ReadVariableOpNmodel_2/model_1/map_features/model/chiral_tag_embedding/BiasAdd/ReadVariableOp2�
Mmodel_2/model_1/map_features/model/chiral_tag_embedding/MatMul/ReadVariableOpMmodel_2/model_1/map_features/model/chiral_tag_embedding/MatMul/ReadVariableOp2�
Dmodel_2/model_1/map_features/model/degree_embedding/embedding_lookupDmodel_2/model_1/map_features/model/degree_embedding/embedding_lookup2�
Kmodel_2/model_1/map_features/model/formal_charge_embedding/embedding_lookupKmodel_2/model_1/map_features/model/formal_charge_embedding/embedding_lookup2�
Qmodel_2/model_1/map_features/model/hybridization_embedding/BiasAdd/ReadVariableOpQmodel_2/model_1/map_features/model/hybridization_embedding/BiasAdd/ReadVariableOp2�
Pmodel_2/model_1/map_features/model/hybridization_embedding/MatMul/ReadVariableOpPmodel_2/model_1/map_features/model/hybridization_embedding/MatMul/ReadVariableOp2�
Imodel_2/model_1/map_features/model/is_aromatic_embedding/embedding_lookupImodel_2/model_1/map_features/model/is_aromatic_embedding/embedding_lookup2�
Imodel_2/model_1/map_features/model/no_implicit_embedding/embedding_lookupImodel_2/model_1/map_features/model/no_implicit_embedding/embedding_lookup2�
Hmodel_2/model_1/map_features/model/node_embedding/BiasAdd/ReadVariableOpHmodel_2/model_1/map_features/model/node_embedding/BiasAdd/ReadVariableOp2�
Gmodel_2/model_1/map_features/model/node_embedding/MatMul/ReadVariableOpGmodel_2/model_1/map_features/model/node_embedding/MatMul/ReadVariableOp2�
Dmodel_2/model_1/map_features/model/num_Hs_embedding/embedding_lookupDmodel_2/model_1/map_features/model/num_Hs_embedding/embedding_lookup2�
Emodel_2/model_1/map_features/model/valence_embedding/embedding_lookupEmodel_2/model_1/map_features/model/valence_embedding/embedding_lookup2�
Omodel_2/model_1/map_features/model_1/bond_type_embedding/BiasAdd/ReadVariableOpOmodel_2/model_1/map_features/model_1/bond_type_embedding/BiasAdd/ReadVariableOp2�
Nmodel_2/model_1/map_features/model_1/bond_type_embedding/MatMul/ReadVariableOpNmodel_2/model_1/map_features/model_1/bond_type_embedding/MatMul/ReadVariableOp2�
Jmodel_2/model_1/map_features/model_1/edge_embedding/BiasAdd/ReadVariableOpJmodel_2/model_1/map_features/model_1/edge_embedding/BiasAdd/ReadVariableOp2�
Imodel_2/model_1/map_features/model_1/edge_embedding/MatMul/ReadVariableOpImodel_2/model_1/map_features/model_1/edge_embedding/MatMul/ReadVariableOp2�
Mmodel_2/model_1/map_features/model_1/is_conjugated_embedding/embedding_lookupMmodel_2/model_1/map_features/model_1/is_conjugated_embedding/embedding_lookup2�
Lmodel_2/model_1/map_features/model_1/stereo_embedding/BiasAdd/ReadVariableOpLmodel_2/model_1/map_features/model_1/stereo_embedding/BiasAdd/ReadVariableOp2�
Kmodel_2/model_1/map_features/model_1/stereo_embedding/MatMul/ReadVariableOpKmodel_2/model_1/map_features/model_1/stereo_embedding/MatMul/ReadVariableOp2�
Kmodel_2/model_1/sequential_29/sequential_27/dense_21/BiasAdd/ReadVariableOpKmodel_2/model_1/sequential_29/sequential_27/dense_21/BiasAdd/ReadVariableOp2�
Jmodel_2/model_1/sequential_29/sequential_27/dense_21/MatMul/ReadVariableOpJmodel_2/model_1/sequential_29/sequential_27/dense_21/MatMul/ReadVariableOp2�
Kmodel_2/model_1/sequential_29/sequential_28/dense_22/BiasAdd/ReadVariableOpKmodel_2/model_1/sequential_29/sequential_28/dense_22/BiasAdd/ReadVariableOp2�
Jmodel_2/model_1/sequential_29/sequential_28/dense_22/MatMul/ReadVariableOpJmodel_2/model_1/sequential_29/sequential_28/dense_22/MatMul/ReadVariableOp2�
?model_2/model_1/structured_readout/assert_equal_1/Assert/Assert?model_2/model_1/structured_readout/assert_equal_1/Assert/Assert2|
<model_2/model_1/structured_readout/assert_less/Assert/Assert<model_2/model_1/structured_readout/assert_less/Assert/Assert2�
bmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assertbmodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert2�
ymodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assertymodel_2/parse_example_2/RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert2�
Hmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/AssertHmodel_2/parse_example_2/RaggedFromRowSplits/assert_equal_1/Assert/Assert2�
dmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assertdmodel_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert2�
{model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert{model_2/parse_example_2/RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert2�
Jmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/AssertJmodel_2/parse_example_2/RaggedFromRowSplits_1/assert_equal_1/Assert/Assert2�
dmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assertdmodel_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert2�
{model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert{model_2/parse_example_2/RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert2�
Jmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/AssertJmodel_2/parse_example_2/RaggedFromRowSplits_2/assert_equal_1/Assert/Assert2�
dmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assertdmodel_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert2�
{model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert{model_2/parse_example_2/RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert2�
Jmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/AssertJmodel_2/parse_example_2/RaggedFromRowSplits_3/assert_equal_1/Assert/Assert2�
dmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assertdmodel_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert2�
{model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert{model_2/parse_example_2/RaggedFromRowSplits_4/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert2�
Jmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/AssertJmodel_2/parse_example_2/RaggedFromRowSplits_4/assert_equal_1/Assert/Assert2�
dmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assertdmodel_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert2�
{model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert{model_2/parse_example_2/RaggedFromRowSplits_5/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert2�
Jmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/AssertJmodel_2/parse_example_2/RaggedFromRowSplits_5/assert_equal_1/Assert/Assert2�
dmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assertdmodel_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_equal_1/Assert/Assert2�
{model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert{model_2/parse_example_2/RaggedFromRowSplits_6/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Assert2�
Jmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/AssertJmodel_2/parse_example_2/RaggedFromRowSplits_6/assert_equal_1/Assert/Assert2j
3model_2/parse_example_2/assert_shapes/Assert/Assert3model_2/parse_example_2/assert_shapes/Assert/Assert2n
5model_2/parse_example_2/assert_shapes/Assert_1/Assert5model_2/parse_example_2/assert_shapes/Assert_1/Assert2n
5model_2/parse_example_2/assert_shapes/Assert_2/Assert5model_2/parse_example_2/assert_shapes/Assert_2/Assert2n
5model_2/parse_example_2/assert_shapes/Assert_3/Assert5model_2/parse_example_2/assert_shapes/Assert_3/Assert:(F$
"
_user_specified_name
resource:(E$
"
_user_specified_name
resource:(D$
"
_user_specified_name
resource:(C$
"
_user_specified_name
resource:(B$
"
_user_specified_name
resource:(A$
"
_user_specified_name
resource:(@$
"
_user_specified_name
resource:(?$
"
_user_specified_name
resource:(>$
"
_user_specified_name
resource:(=$
"
_user_specified_name
resource:(<$
"
_user_specified_name
resource:(;$
"
_user_specified_name
resource:(:$
"
_user_specified_name
resource:(9$
"
_user_specified_name
resource:(8$
"
_user_specified_name
resource:(7$
"
_user_specified_name
resource:(6$
"
_user_specified_name
resource:(5$
"
_user_specified_name
resource:(4$
"
_user_specified_name
resource:(3$
"
_user_specified_name
resource:(2$
"
_user_specified_name
resource:(1$
"
_user_specified_name
resource:(0$
"
_user_specified_name
resource:(/$
"
_user_specified_name
resource:(.$
"
_user_specified_name
resource:(-$
"
_user_specified_name
resource:(,$
"
_user_specified_name
resource:(+$
"
_user_specified_name
resource:(*$
"
_user_specified_name
resource:()$
"
_user_specified_name
resource:(($
"
_user_specified_name
resource:('$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:&"
 
_user_specified_name315427:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:&"
 
_user_specified_name315391:&"
 
_user_specified_name315387:&
"
 
_user_specified_name315383:&	"
 
_user_specified_name315379:&"
 
_user_specified_name315375:&"
 
_user_specified_name315371:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:M I
#
_output_shapes
:���������
"
_user_specified_name
examples
��
�?
__inference__traced_save_316894
file_prefixB
0read_disablecopyonread_atom_sym_embedding_kernel:>
0read_1_disablecopyonread_atom_sym_embedding_bias:F
4read_2_disablecopyonread_chiral_tag_embedding_kernel:	@
2read_3_disablecopyonread_chiral_tag_embedding_bias:I
7read_4_disablecopyonread_hybridization_embedding_kernel:	C
5read_5_disablecopyonread_hybridization_embedding_bias:F
4read_6_disablecopyonread_degree_embedding_embeddings:M
;read_7_disablecopyonread_formal_charge_embedding_embeddings:K
9read_8_disablecopyonread_is_aromatic_embedding_embeddings:K
9read_9_disablecopyonread_no_implicit_embedding_embeddings:G
5read_10_disablecopyonread_num_hs_embedding_embeddings:H
6read_11_disablecopyonread_valence_embedding_embeddings:B
/read_12_disablecopyonread_node_embedding_kernel:	�<
-read_13_disablecopyonread_node_embedding_bias:	�F
4read_14_disablecopyonread_bond_type_embedding_kernel:@
2read_15_disablecopyonread_bond_type_embedding_bias:N
<read_16_disablecopyonread_is_conjugated_embedding_embeddings:C
1read_17_disablecopyonread_stereo_embedding_kernel:=
/read_18_disablecopyonread_stereo_embedding_bias:B
/read_19_disablecopyonread_edge_embedding_kernel:	�<
-read_20_disablecopyonread_edge_embedding_bias:	�:
&read_21_disablecopyonread_dense_kernel:
��3
$read_22_disablecopyonread_dense_bias:	�<
(read_23_disablecopyonread_dense_1_kernel:
��5
&read_24_disablecopyonread_dense_1_bias:	�<
(read_25_disablecopyonread_dense_2_kernel:
��5
&read_26_disablecopyonread_dense_2_bias:	�<
(read_27_disablecopyonread_dense_3_kernel:
��5
&read_28_disablecopyonread_dense_3_bias:	�<
(read_29_disablecopyonread_dense_4_kernel:
��5
&read_30_disablecopyonread_dense_4_bias:	�<
(read_31_disablecopyonread_dense_5_kernel:
��5
&read_32_disablecopyonread_dense_5_bias:	�<
(read_33_disablecopyonread_dense_6_kernel:
��5
&read_34_disablecopyonread_dense_6_bias:	�<
(read_35_disablecopyonread_dense_7_kernel:
��5
&read_36_disablecopyonread_dense_7_bias:	�<
(read_37_disablecopyonread_dense_8_kernel:
��5
&read_38_disablecopyonread_dense_8_bias:	�<
(read_39_disablecopyonread_dense_9_kernel:
��5
&read_40_disablecopyonread_dense_9_bias:	�=
)read_41_disablecopyonread_dense_10_kernel:
��6
'read_42_disablecopyonread_dense_10_bias:	�=
)read_43_disablecopyonread_dense_11_kernel:
��6
'read_44_disablecopyonread_dense_11_bias:	�=
)read_45_disablecopyonread_dense_12_kernel:
��6
'read_46_disablecopyonread_dense_12_bias:	�=
)read_47_disablecopyonread_dense_13_kernel:
��6
'read_48_disablecopyonread_dense_13_bias:	�=
)read_49_disablecopyonread_dense_14_kernel:
��6
'read_50_disablecopyonread_dense_14_bias:	�=
)read_51_disablecopyonread_dense_15_kernel:
��6
'read_52_disablecopyonread_dense_15_bias:	�=
)read_53_disablecopyonread_dense_16_kernel:
��6
'read_54_disablecopyonread_dense_16_bias:	�=
)read_55_disablecopyonread_dense_17_kernel:
��6
'read_56_disablecopyonread_dense_17_bias:	�=
)read_57_disablecopyonread_dense_18_kernel:
��6
'read_58_disablecopyonread_dense_18_bias:	�=
)read_59_disablecopyonread_dense_19_kernel:
��6
'read_60_disablecopyonread_dense_19_bias:	�=
)read_61_disablecopyonread_dense_20_kernel:
��6
'read_62_disablecopyonread_dense_20_bias:	�=
)read_63_disablecopyonread_dense_21_kernel:
��6
'read_64_disablecopyonread_dense_21_bias:	�=
)read_65_disablecopyonread_dense_22_kernel:
��6
'read_66_disablecopyonread_dense_22_bias:	�@
-read_67_disablecopyonread_gru_gru_cell_kernel:	�I
7read_68_disablecopyonread_gru_gru_cell_recurrent_kernel:=
+read_69_disablecopyonread_gru_gru_cell_bias:
savev2_const
identity_141��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead0read_disablecopyonread_atom_sym_embedding_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp0read_disablecopyonread_atom_sym_embedding_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_1/DisableCopyOnReadDisableCopyOnRead0read_1_disablecopyonread_atom_sym_embedding_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp0read_1_disablecopyonread_atom_sym_embedding_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_2/DisableCopyOnReadDisableCopyOnRead4read_2_disablecopyonread_chiral_tag_embedding_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp4read_2_disablecopyonread_chiral_tag_embedding_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:	*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:	c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:	�
Read_3/DisableCopyOnReadDisableCopyOnRead2read_3_disablecopyonread_chiral_tag_embedding_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp2read_3_disablecopyonread_chiral_tag_embedding_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_4/DisableCopyOnReadDisableCopyOnRead7read_4_disablecopyonread_hybridization_embedding_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp7read_4_disablecopyonread_hybridization_embedding_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:	*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:	c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:	�
Read_5/DisableCopyOnReadDisableCopyOnRead5read_5_disablecopyonread_hybridization_embedding_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp5read_5_disablecopyonread_hybridization_embedding_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_6/DisableCopyOnReadDisableCopyOnRead4read_6_disablecopyonread_degree_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp4read_6_disablecopyonread_degree_embedding_embeddings^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_7/DisableCopyOnReadDisableCopyOnRead;read_7_disablecopyonread_formal_charge_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp;read_7_disablecopyonread_formal_charge_embedding_embeddings^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_8/DisableCopyOnReadDisableCopyOnRead9read_8_disablecopyonread_is_aromatic_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp9read_8_disablecopyonread_is_aromatic_embedding_embeddings^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_9/DisableCopyOnReadDisableCopyOnRead9read_9_disablecopyonread_no_implicit_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp9read_9_disablecopyonread_no_implicit_embedding_embeddings^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_10/DisableCopyOnReadDisableCopyOnRead5read_10_disablecopyonread_num_hs_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp5read_10_disablecopyonread_num_hs_embedding_embeddings^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_11/DisableCopyOnReadDisableCopyOnRead6read_11_disablecopyonread_valence_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp6read_11_disablecopyonread_valence_embedding_embeddings^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_12/DisableCopyOnReadDisableCopyOnRead/read_12_disablecopyonread_node_embedding_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp/read_12_disablecopyonread_node_embedding_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_13/DisableCopyOnReadDisableCopyOnRead-read_13_disablecopyonread_node_embedding_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp-read_13_disablecopyonread_node_embedding_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_14/DisableCopyOnReadDisableCopyOnRead4read_14_disablecopyonread_bond_type_embedding_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp4read_14_disablecopyonread_bond_type_embedding_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_15/DisableCopyOnReadDisableCopyOnRead2read_15_disablecopyonread_bond_type_embedding_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp2read_15_disablecopyonread_bond_type_embedding_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnRead<read_16_disablecopyonread_is_conjugated_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp<read_16_disablecopyonread_is_conjugated_embedding_embeddings^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_17/DisableCopyOnReadDisableCopyOnRead1read_17_disablecopyonread_stereo_embedding_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp1read_17_disablecopyonread_stereo_embedding_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_18/DisableCopyOnReadDisableCopyOnRead/read_18_disablecopyonread_stereo_embedding_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp/read_18_disablecopyonread_stereo_embedding_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnRead/read_19_disablecopyonread_edge_embedding_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp/read_19_disablecopyonread_edge_embedding_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_20/DisableCopyOnReadDisableCopyOnRead-read_20_disablecopyonread_edge_embedding_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp-read_20_disablecopyonread_edge_embedding_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_21/DisableCopyOnReadDisableCopyOnRead&read_21_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp&read_21_disablecopyonread_dense_kernel^Read_21/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_22/DisableCopyOnReadDisableCopyOnRead$read_22_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp$read_22_disablecopyonread_dense_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_23/DisableCopyOnReadDisableCopyOnRead(read_23_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp(read_23_disablecopyonread_dense_1_kernel^Read_23/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_24/DisableCopyOnReadDisableCopyOnRead&read_24_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp&read_24_disablecopyonread_dense_1_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_25/DisableCopyOnReadDisableCopyOnRead(read_25_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp(read_25_disablecopyonread_dense_2_kernel^Read_25/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_26/DisableCopyOnReadDisableCopyOnRead&read_26_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp&read_26_disablecopyonread_dense_2_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_27/DisableCopyOnReadDisableCopyOnRead(read_27_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp(read_27_disablecopyonread_dense_3_kernel^Read_27/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_28/DisableCopyOnReadDisableCopyOnRead&read_28_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp&read_28_disablecopyonread_dense_3_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_29/DisableCopyOnReadDisableCopyOnRead(read_29_disablecopyonread_dense_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp(read_29_disablecopyonread_dense_4_kernel^Read_29/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_30/DisableCopyOnReadDisableCopyOnRead&read_30_disablecopyonread_dense_4_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp&read_30_disablecopyonread_dense_4_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_31/DisableCopyOnReadDisableCopyOnRead(read_31_disablecopyonread_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp(read_31_disablecopyonread_dense_5_kernel^Read_31/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_32/DisableCopyOnReadDisableCopyOnRead&read_32_disablecopyonread_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp&read_32_disablecopyonread_dense_5_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_33/DisableCopyOnReadDisableCopyOnRead(read_33_disablecopyonread_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp(read_33_disablecopyonread_dense_6_kernel^Read_33/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_34/DisableCopyOnReadDisableCopyOnRead&read_34_disablecopyonread_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp&read_34_disablecopyonread_dense_6_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_35/DisableCopyOnReadDisableCopyOnRead(read_35_disablecopyonread_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp(read_35_disablecopyonread_dense_7_kernel^Read_35/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_36/DisableCopyOnReadDisableCopyOnRead&read_36_disablecopyonread_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp&read_36_disablecopyonread_dense_7_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_37/DisableCopyOnReadDisableCopyOnRead(read_37_disablecopyonread_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp(read_37_disablecopyonread_dense_8_kernel^Read_37/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_38/DisableCopyOnReadDisableCopyOnRead&read_38_disablecopyonread_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp&read_38_disablecopyonread_dense_8_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_39/DisableCopyOnReadDisableCopyOnRead(read_39_disablecopyonread_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp(read_39_disablecopyonread_dense_9_kernel^Read_39/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_40/DisableCopyOnReadDisableCopyOnRead&read_40_disablecopyonread_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp&read_40_disablecopyonread_dense_9_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_41/DisableCopyOnReadDisableCopyOnRead)read_41_disablecopyonread_dense_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp)read_41_disablecopyonread_dense_10_kernel^Read_41/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_42/DisableCopyOnReadDisableCopyOnRead'read_42_disablecopyonread_dense_10_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp'read_42_disablecopyonread_dense_10_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_43/DisableCopyOnReadDisableCopyOnRead)read_43_disablecopyonread_dense_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp)read_43_disablecopyonread_dense_11_kernel^Read_43/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_44/DisableCopyOnReadDisableCopyOnRead'read_44_disablecopyonread_dense_11_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp'read_44_disablecopyonread_dense_11_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_45/DisableCopyOnReadDisableCopyOnRead)read_45_disablecopyonread_dense_12_kernel"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp)read_45_disablecopyonread_dense_12_kernel^Read_45/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_46/DisableCopyOnReadDisableCopyOnRead'read_46_disablecopyonread_dense_12_bias"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp'read_46_disablecopyonread_dense_12_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_47/DisableCopyOnReadDisableCopyOnRead)read_47_disablecopyonread_dense_13_kernel"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp)read_47_disablecopyonread_dense_13_kernel^Read_47/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_48/DisableCopyOnReadDisableCopyOnRead'read_48_disablecopyonread_dense_13_bias"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp'read_48_disablecopyonread_dense_13_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_49/DisableCopyOnReadDisableCopyOnRead)read_49_disablecopyonread_dense_14_kernel"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp)read_49_disablecopyonread_dense_14_kernel^Read_49/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_50/DisableCopyOnReadDisableCopyOnRead'read_50_disablecopyonread_dense_14_bias"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp'read_50_disablecopyonread_dense_14_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_51/DisableCopyOnReadDisableCopyOnRead)read_51_disablecopyonread_dense_15_kernel"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp)read_51_disablecopyonread_dense_15_kernel^Read_51/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_52/DisableCopyOnReadDisableCopyOnRead'read_52_disablecopyonread_dense_15_bias"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp'read_52_disablecopyonread_dense_15_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_53/DisableCopyOnReadDisableCopyOnRead)read_53_disablecopyonread_dense_16_kernel"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp)read_53_disablecopyonread_dense_16_kernel^Read_53/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_54/DisableCopyOnReadDisableCopyOnRead'read_54_disablecopyonread_dense_16_bias"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp'read_54_disablecopyonread_dense_16_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_55/DisableCopyOnReadDisableCopyOnRead)read_55_disablecopyonread_dense_17_kernel"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp)read_55_disablecopyonread_dense_17_kernel^Read_55/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_56/DisableCopyOnReadDisableCopyOnRead'read_56_disablecopyonread_dense_17_bias"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp'read_56_disablecopyonread_dense_17_bias^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_57/DisableCopyOnReadDisableCopyOnRead)read_57_disablecopyonread_dense_18_kernel"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp)read_57_disablecopyonread_dense_18_kernel^Read_57/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_58/DisableCopyOnReadDisableCopyOnRead'read_58_disablecopyonread_dense_18_bias"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp'read_58_disablecopyonread_dense_18_bias^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_59/DisableCopyOnReadDisableCopyOnRead)read_59_disablecopyonread_dense_19_kernel"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp)read_59_disablecopyonread_dense_19_kernel^Read_59/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_60/DisableCopyOnReadDisableCopyOnRead'read_60_disablecopyonread_dense_19_bias"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp'read_60_disablecopyonread_dense_19_bias^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_61/DisableCopyOnReadDisableCopyOnRead)read_61_disablecopyonread_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp)read_61_disablecopyonread_dense_20_kernel^Read_61/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_62/DisableCopyOnReadDisableCopyOnRead'read_62_disablecopyonread_dense_20_bias"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp'read_62_disablecopyonread_dense_20_bias^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_63/DisableCopyOnReadDisableCopyOnRead)read_63_disablecopyonread_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp)read_63_disablecopyonread_dense_21_kernel^Read_63/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_64/DisableCopyOnReadDisableCopyOnRead'read_64_disablecopyonread_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp'read_64_disablecopyonread_dense_21_bias^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_65/DisableCopyOnReadDisableCopyOnRead)read_65_disablecopyonread_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp)read_65_disablecopyonread_dense_22_kernel^Read_65/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_66/DisableCopyOnReadDisableCopyOnRead'read_66_disablecopyonread_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp'read_66_disablecopyonread_dense_22_bias^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_67/DisableCopyOnReadDisableCopyOnRead-read_67_disablecopyonread_gru_gru_cell_kernel"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp-read_67_disablecopyonread_gru_gru_cell_kernel^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_68/DisableCopyOnReadDisableCopyOnRead7read_68_disablecopyonread_gru_gru_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp7read_68_disablecopyonread_gru_gru_cell_recurrent_kernel^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_69/DisableCopyOnReadDisableCopyOnRead+read_69_disablecopyonread_gru_gru_cell_bias"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp+read_69_disablecopyonread_gru_gru_cell_bias^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes

:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*�
value�B�GB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB'variables/62/.ATTRIBUTES/VARIABLE_VALUEB'variables/63/.ATTRIBUTES/VARIABLE_VALUEB'variables/64/.ATTRIBUTES/VARIABLE_VALUEB'variables/65/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/68/.ATTRIBUTES/VARIABLE_VALUEB'variables/69/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*�
value�B�GB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *U
dtypesK
I2G�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_140Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_141IdentityIdentity_140:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_141Identity_141:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=G9

_output_shapes
: 

_user_specified_nameConst:1F-
+
_user_specified_namegru/gru_cell/bias:=E9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3D/
-
_user_specified_namegru/gru_cell/kernel:-C)
'
_user_specified_namedense_22/bias:/B+
)
_user_specified_namedense_22/kernel:-A)
'
_user_specified_namedense_21/bias:/@+
)
_user_specified_namedense_21/kernel:-?)
'
_user_specified_namedense_20/bias:/>+
)
_user_specified_namedense_20/kernel:-=)
'
_user_specified_namedense_19/bias:/<+
)
_user_specified_namedense_19/kernel:-;)
'
_user_specified_namedense_18/bias:/:+
)
_user_specified_namedense_18/kernel:-9)
'
_user_specified_namedense_17/bias:/8+
)
_user_specified_namedense_17/kernel:-7)
'
_user_specified_namedense_16/bias:/6+
)
_user_specified_namedense_16/kernel:-5)
'
_user_specified_namedense_15/bias:/4+
)
_user_specified_namedense_15/kernel:-3)
'
_user_specified_namedense_14/bias:/2+
)
_user_specified_namedense_14/kernel:-1)
'
_user_specified_namedense_13/bias:/0+
)
_user_specified_namedense_13/kernel:-/)
'
_user_specified_namedense_12/bias:/.+
)
_user_specified_namedense_12/kernel:--)
'
_user_specified_namedense_11/bias:/,+
)
_user_specified_namedense_11/kernel:-+)
'
_user_specified_namedense_10/bias:/*+
)
_user_specified_namedense_10/kernel:,)(
&
_user_specified_namedense_9/bias:.(*
(
_user_specified_namedense_9/kernel:,'(
&
_user_specified_namedense_8/bias:.&*
(
_user_specified_namedense_8/kernel:,%(
&
_user_specified_namedense_7/bias:.$*
(
_user_specified_namedense_7/kernel:,#(
&
_user_specified_namedense_6/bias:."*
(
_user_specified_namedense_6/kernel:,!(
&
_user_specified_namedense_5/bias:. *
(
_user_specified_namedense_5/kernel:,(
&
_user_specified_namedense_4/bias:.*
(
_user_specified_namedense_4/kernel:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:3/
-
_user_specified_nameedge_embedding/bias:51
/
_user_specified_nameedge_embedding/kernel:51
/
_user_specified_namestereo_embedding/bias:73
1
_user_specified_namestereo_embedding/kernel:B>
<
_user_specified_name$"is_conjugated_embedding/embeddings:84
2
_user_specified_namebond_type_embedding/bias::6
4
_user_specified_namebond_type_embedding/kernel:3/
-
_user_specified_namenode_embedding/bias:51
/
_user_specified_namenode_embedding/kernel:<8
6
_user_specified_namevalence_embedding/embeddings:;7
5
_user_specified_namenum_Hs_embedding/embeddings:@
<
:
_user_specified_name" no_implicit_embedding/embeddings:@	<
:
_user_specified_name" is_aromatic_embedding/embeddings:B>
<
_user_specified_name$"formal_charge_embedding/embeddings:;7
5
_user_specified_namedegree_embedding/embeddings:<8
6
_user_specified_namehybridization_embedding/bias:>:
8
_user_specified_name hybridization_embedding/kernel:95
3
_user_specified_namechiral_tag_embedding/bias:;7
5
_user_specified_namechiral_tag_embedding/kernel:73
1
_user_specified_nameatom_sym_embedding/bias:95
3
_user_specified_nameatom_sym_embedding/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
while_cond_315853
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_315853___redundant_placeholder04
0while_while_cond_315853___redundant_placeholder14
0while_while_cond_315853___redundant_placeholder24
0while_while_cond_315853___redundant_placeholder34
0while_while_cond_315853___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :���������: ::::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�4
�
-__inference_signature_wrapper___call___316305
examples
unknown:
	unknown_0:
	unknown_1:	
	unknown_2:
	unknown_3:	
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:	�

unknown_12:	�

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:	�

unknown_19:	�

unknown_20:
��

unknown_21:	�

unknown_22:
��

unknown_23:	�

unknown_24:
��

unknown_25:	�

unknown_26:
��

unknown_27:	�

unknown_28:
��

unknown_29:	�

unknown_30:
��

unknown_31:	�

unknown_32:
��

unknown_33:	�

unknown_34:
��

unknown_35:	�

unknown_36:
��

unknown_37:	�

unknown_38:
��

unknown_39:	�

unknown_40:
��

unknown_41:	�

unknown_42:
��

unknown_43:	�

unknown_44:
��

unknown_45:	�

unknown_46:
��

unknown_47:	�

unknown_48:
��

unknown_49:	�

unknown_50:
��

unknown_51:	�

unknown_52:
��

unknown_53:	�

unknown_54:
��

unknown_55:	�

unknown_56:
��

unknown_57:	�

unknown_58:
��

unknown_59:	�

unknown_60:
��

unknown_61:	�

unknown_62:
��

unknown_63:	�

unknown_64:
��

unknown_65:	�

unknown_66:	�

unknown_67:

unknown_68:
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*h
_read_only_resource_inputsJ
HF	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEF*-
config_proto

CPU

GPU 2J 8� *$
fR
__inference___call___316159s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&F"
 
_user_specified_name316301:&E"
 
_user_specified_name316299:&D"
 
_user_specified_name316297:&C"
 
_user_specified_name316295:&B"
 
_user_specified_name316293:&A"
 
_user_specified_name316291:&@"
 
_user_specified_name316289:&?"
 
_user_specified_name316287:&>"
 
_user_specified_name316285:&="
 
_user_specified_name316283:&<"
 
_user_specified_name316281:&;"
 
_user_specified_name316279:&:"
 
_user_specified_name316277:&9"
 
_user_specified_name316275:&8"
 
_user_specified_name316273:&7"
 
_user_specified_name316271:&6"
 
_user_specified_name316269:&5"
 
_user_specified_name316267:&4"
 
_user_specified_name316265:&3"
 
_user_specified_name316263:&2"
 
_user_specified_name316261:&1"
 
_user_specified_name316259:&0"
 
_user_specified_name316257:&/"
 
_user_specified_name316255:&."
 
_user_specified_name316253:&-"
 
_user_specified_name316251:&,"
 
_user_specified_name316249:&+"
 
_user_specified_name316247:&*"
 
_user_specified_name316245:&)"
 
_user_specified_name316243:&("
 
_user_specified_name316241:&'"
 
_user_specified_name316239:&&"
 
_user_specified_name316237:&%"
 
_user_specified_name316235:&$"
 
_user_specified_name316233:&#"
 
_user_specified_name316231:&""
 
_user_specified_name316229:&!"
 
_user_specified_name316227:& "
 
_user_specified_name316225:&"
 
_user_specified_name316223:&"
 
_user_specified_name316221:&"
 
_user_specified_name316219:&"
 
_user_specified_name316217:&"
 
_user_specified_name316215:&"
 
_user_specified_name316213:&"
 
_user_specified_name316211:&"
 
_user_specified_name316209:&"
 
_user_specified_name316207:&"
 
_user_specified_name316205:&"
 
_user_specified_name316203:&"
 
_user_specified_name316201:&"
 
_user_specified_name316199:&"
 
_user_specified_name316197:&"
 
_user_specified_name316195:&"
 
_user_specified_name316193:&"
 
_user_specified_name316191:&"
 
_user_specified_name316189:&"
 
_user_specified_name316187:&"
 
_user_specified_name316185:&"
 
_user_specified_name316183:&
"
 
_user_specified_name316181:&	"
 
_user_specified_name316179:&"
 
_user_specified_name316177:&"
 
_user_specified_name316175:&"
 
_user_specified_name316173:&"
 
_user_specified_name316171:&"
 
_user_specified_name316169:&"
 
_user_specified_name316167:&"
 
_user_specified_name316165:&"
 
_user_specified_name316163:M I
#
_output_shapes
:���������
"
_user_specified_name
examples
�/
�
while_body_315854
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:���������{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:���������W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split�
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:����������
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:���������Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:���������Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:���������t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:���������]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:���������o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:���������k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:���������U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:���������l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:���������c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:���������h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:���������"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������: : :	�::::C
?

_output_shapes
:
!
_user_specified_name	unstack:P	L

_output_shapes

:
*
_user_specified_namerecurrent_kernel:C?

_output_shapes
:
!
_user_specified_name	unstack:GC

_output_shapes
:	�
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�>
�
&__forward_gpu_gru_with_fallback_316155

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	�:	�:	�*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:::*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������U
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes
:0S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	�[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	�[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	�[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:Z
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:Z
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes
:@a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:Z
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes
:@[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:���������:���������: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:���������p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:���������*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:���������]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:���������Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:���������I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "#
concat_axisconcat/axis:output:0"
concatconcat_0:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"
cudnnrnnCudnnRNN:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:����������:���������:	�::*<
api_implements*(gru_95a99024-6e46-4256-a2c3-70de478e526d*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_gpu_gru_with_fallback_316020_316156*
go_backwards( *

time_major( :D@

_output_shapes

:

_user_specified_namebias:PL

_output_shapes

:
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	�
 
_user_specified_namekernel:OK
'
_output_shapes
:���������
 
_user_specified_nameinit_h:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�

:__inference___backward_gpu_gru_with_fallback_316020_316156
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:���������d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:���������`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:���������O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::���
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:���������*
shrink_axis_mask�
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:�
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::���
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:����������
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:���������a
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*L
_output_shapes:
8:����������:���������: :�1*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:�����������
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::���
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:���������\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�g
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:@g
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:@g
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:@g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:��
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:@�
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes
:@�
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes
:@�
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:�
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:�
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:�
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:�
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:�
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	�o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	�o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	�o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:�
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	��
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	��
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	��
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:�
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:�
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:�
gradients/split_2_grad/concatConcatV2)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes
:0�
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	��
gradients/split_1_grad/concatConcatV2(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:s
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:����������t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:���������f

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	�g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������: :���������::���������: ::����������:���������: :�1::���������: ::::::: : : *<
api_implements*(gru_95a99024-6e46-4256-a2c3-70de478e526d*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_gru_with_fallback_316155*
go_backwards( *

time_major( :IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:OK
'
_output_shapes
:���������
 
_user_specified_nameinit_h:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:C?

_output_shapes	
:�1
 
_user_specified_nameconcat:HD

_output_shapes
: 
*
_user_specified_nameCudnnRNN/input_c:W
S
+
_output_shapes
:���������
$
_user_specified_name
ExpandDims:W	S
,
_output_shapes
:����������
#
_user_specified_name	transpose:B>

_output_shapes
:
"
_user_specified_name
CudnnRNN:@<

_output_shapes
: 
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:���������
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:UQ
+
_output_shapes
:���������
"
_user_specified_name
CudnnRNN:

_output_shapes
: :-)
'
_output_shapes
:���������:1-
+
_output_shapes
:���������:- )
'
_output_shapes
:���������
�<
�
__inference_standard_gru_315943

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
::*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:����������P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:���������h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:���������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:���������l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:���������S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:���������:���������:���������*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:���������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*T
_output_shapesB
@: : : : :���������: : :	�:::* 
_read_only_resource_inputs
 *
bodyR
while_body_315854*
condR
while_cond_315853*S
output_shapesB
@: : : : :���������: : :	�:::*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:���������]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:���������X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:���������I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:����������:���������:	�::*<
api_implements*(gru_95a99024-6e46-4256-a2c3-70de478e526d*
api_preferred_deviceCPU*
go_backwards( *

time_major( :D@

_output_shapes

:

_user_specified_namebias:PL

_output_shapes

:
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	�
 
_user_specified_namekernel:OK
'
_output_shapes
:���������
 
_user_specified_nameinit_h:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serve�
/
examples#
serve_examples:0���������=
shape4
StatefulPartitionedCall:0���������tensorflow/serving/predict*�
serving_default�
9
examples-
serving_default_examples:0���������?
shape6
StatefulPartitionedCall_1:0���������tensorflow/serving/predict:�1
�
_endpoint_names
_endpoint_signatures
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve
	
signatures"
_generic_user_object
 "
trackable_list_wrapper
+
	
serve"
trackable_dict_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
 21
!22
"23
#24
$25
%26
&27
'28
(29
)30
*31
+32
,33
-34
.35
/36
037
138
239
340
441
542
643
744
845
946
:47
;48
<49
=50
>51
?52
@53
A54
B55
C56
D57
E58
F59
G60
H61
I62
J63
K64
L65
M66
N67
O68
P69"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
 21
!22
"23
#24
$25
%26
&27
'28
(29
)30
*31
+32
,33
-34
.35
/36
037
138
239
340
441
542
643
744
845
946
:47
;48
<49
=50
>51
?52
@53
A54
B55
C56
D57
E58
F59
G60
H61
I62
J63
K64
L65
M66
N67
O68
P69"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
14
65
96
E7
N8
9
10
#11
)12
-13
714
;15
B16
I17
O18
19
 20
!21
$22
/23
324
C25
K26
P27
28
29
30
31
&32
833
<34
35
36
%37
=38
@39
H40
J41
L42
43
044
245
:46
>47
G48
M49
50
51
52
"53
'54
(55
,56
457
?58
F59
60
61
62
63
*64
+65
.66
567
A68
D69"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Qtrace_02�
__inference___call___316159�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *#� 
�
examples���������zQtrace_0
7
	Rserve
Sserving_default"
signature_map
 "
trackable_list_wrapper
+:)2atom_sym_embedding/kernel
%:#2atom_sym_embedding/bias
-:+	2chiral_tag_embedding/kernel
':%2chiral_tag_embedding/bias
0:.	2hybridization_embedding/kernel
*:(2hybridization_embedding/bias
-:+2degree_embedding/embeddings
4:22"formal_charge_embedding/embeddings
2:02 is_aromatic_embedding/embeddings
2:02 no_implicit_embedding/embeddings
-:+2num_Hs_embedding/embeddings
.:,2valence_embedding/embeddings
(:&	�2node_embedding/kernel
": �2node_embedding/bias
,:*2bond_type_embedding/kernel
&:$2bond_type_embedding/bias
4:22"is_conjugated_embedding/embeddings
):'2stereo_embedding/kernel
#:!2stereo_embedding/bias
(:&	�2edge_embedding/kernel
": �2edge_embedding/bias
 :
��2dense/kernel
:�2
dense/bias
": 
��2dense_1/kernel
:�2dense_1/bias
": 
��2dense_2/kernel
:�2dense_2/bias
": 
��2dense_3/kernel
:�2dense_3/bias
": 
��2dense_4/kernel
:�2dense_4/bias
": 
��2dense_5/kernel
:�2dense_5/bias
": 
��2dense_6/kernel
:�2dense_6/bias
": 
��2dense_7/kernel
:�2dense_7/bias
": 
��2dense_8/kernel
:�2dense_8/bias
": 
��2dense_9/kernel
:�2dense_9/bias
#:!
��2dense_10/kernel
:�2dense_10/bias
#:!
��2dense_11/kernel
:�2dense_11/bias
#:!
��2dense_12/kernel
:�2dense_12/bias
#:!
��2dense_13/kernel
:�2dense_13/bias
#:!
��2dense_14/kernel
:�2dense_14/bias
#:!
��2dense_15/kernel
:�2dense_15/bias
#:!
��2dense_16/kernel
:�2dense_16/bias
#:!
��2dense_17/kernel
:�2dense_17/bias
#:!
��2dense_18/kernel
:�2dense_18/bias
#:!
��2dense_19/kernel
:�2dense_19/bias
#:!
��2dense_20/kernel
:�2dense_20/bias
#:!
��2dense_21/kernel
:�2dense_21/bias
#:!
��2dense_22/kernel
:�2dense_22/bias
&:$	�2gru/gru_cell/kernel
/:-2gru/gru_cell/recurrent_kernel
#:!2gru/gru_cell/bias
�B�
__inference___call___316159examples"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_signature_wrapper___call___316305examples"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jexamples
kwonlydefaults
 
annotations� *
 
�B�
-__inference_signature_wrapper___call___316450examples"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jexamples
kwonlydefaults
 
annotations� *
 �
__inference___call___316159�F !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOP-�*
#� 
�
examples���������
� "1�.
,
shape#� 
shape����������
-__inference_signature_wrapper___call___316305�F !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOP9�6
� 
/�,
*
examples�
examples���������"1�.
,
shape#� 
shape����������
-__inference_signature_wrapper___call___316450�F !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOP9�6
� 
/�,
*
examples�
examples���������"1�.
,
shape#� 
shape���������