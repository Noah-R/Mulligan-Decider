??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
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
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
?
sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_namesequential/dense/kernel
?
+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel*
_output_shapes

:
*
dtype0
?
sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namesequential/dense/bias
{
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
#RMSprop/sequential/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*4
shared_name%#RMSprop/sequential/dense/kernel/rms
?
7RMSprop/sequential/dense/kernel/rms/Read/ReadVariableOpReadVariableOp#RMSprop/sequential/dense/kernel/rms*
_output_shapes

:
*
dtype0
?
!RMSprop/sequential/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!RMSprop/sequential/dense/bias/rms
?
5RMSprop/sequential/dense/bias/rms/Read/ReadVariableOpReadVariableOp!RMSprop/sequential/dense/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
	optimizer
_build_input_shape
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
x

_feature_columns

_resources
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
U
iter
	decay
learning_rate
momentum
rho	rms;	rms<
 

0
1

0
1
 
?
trainable_variables
	variables

layers
regularization_losses
non_trainable_variables
layer_regularization_losses
metrics
layer_metrics
 
 
 
 
 
 
?
trainable_variables
 metrics

!layers
	variables
regularization_losses
"layer_regularization_losses
#non_trainable_variables
$layer_metrics
ca
VARIABLE_VALUEsequential/dense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEsequential/dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
%metrics

&layers
	variables
regularization_losses
'layer_regularization_losses
(non_trainable_variables
)layer_metrics
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 

*0
+1
,2
 
 
 
 
 
 
 
 
 
 
 
4
	-total
	.count
/	variables
0	keras_api
D
	1total
	2count
3
_fn_kwargs
4	variables
5	keras_api
D
	6total
	7count
8
_fn_kwargs
9	variables
:	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

/	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

4	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

9	variables
??
VARIABLE_VALUE#RMSprop/sequential/dense/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!RMSprop/sequential/dense/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
x
serving_default_cardsPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
~
serving_default_five_landerPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
~
serving_default_four_landerPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
x
serving_default_foursPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
z
serving_default_on_playPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
w
serving_default_onesPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????

serving_default_three_landerPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
y
serving_default_threesPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
}
serving_default_two_landerPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
w
serving_default_twosPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_cardsserving_default_five_landerserving_default_four_landerserving_default_foursserving_default_on_playserving_default_onesserving_default_three_landerserving_default_threesserving_default_two_landerserving_default_twossequential/dense/kernelsequential/dense/bias*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_11236355
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+sequential/dense/kernel/Read/ReadVariableOp)sequential/dense/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp7RMSprop/sequential/dense/kernel/rms/Read/ReadVariableOp5RMSprop/sequential/dense/bias/rms/Read/ReadVariableOpConst*
Tin
2	*
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
GPU 2J 8? **
f%R#
!__inference__traced_save_11236950
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential/dense/kernelsequential/dense/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1total_2count_2#RMSprop/sequential/dense/kernel/rms!RMSprop/sequential/dense/bias/rms*
Tin
2*
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_11237005??

?
?
H__inference_sequential_layer_call_and_return_conditional_losses_11236266

inputs	
inputs_1	
inputs_2	
inputs_3	
inputs_4	
inputs_5	
inputs_6	
inputs_7	
inputs_8	
inputs_9	 
dense_11236260:

dense_11236262:
identity??dense/StatefulPartitionedCall?
dense_features/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9*
Tin
2
										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_dense_features_layer_call_and_return_conditional_losses_112362152 
dense_features/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0dense_11236260dense_11236262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_112360602
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityn
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_layer_call_and_return_conditional_losses_11236579
inputs_cards	
inputs_five_lander	
inputs_four_lander	
inputs_fours	
inputs_on_play	
inputs_ones	
inputs_three_lander	
inputs_threes	
inputs_two_lander	
inputs_twos	6
$dense_matmul_readvariableop_resource:
3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense_features/cards/CastCastinputs_cards*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
dense_features/cards/Cast?
dense_features/cards/ShapeShapedense_features/cards/Cast:y:0*
T0*
_output_shapes
:2
dense_features/cards/Shape?
(dense_features/cards/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(dense_features/cards/strided_slice/stack?
*dense_features/cards/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features/cards/strided_slice/stack_1?
*dense_features/cards/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features/cards/strided_slice/stack_2?
"dense_features/cards/strided_sliceStridedSlice#dense_features/cards/Shape:output:01dense_features/cards/strided_slice/stack:output:03dense_features/cards/strided_slice/stack_1:output:03dense_features/cards/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"dense_features/cards/strided_slice?
$dense_features/cards/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$dense_features/cards/Reshape/shape/1?
"dense_features/cards/Reshape/shapePack+dense_features/cards/strided_slice:output:0-dense_features/cards/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2$
"dense_features/cards/Reshape/shape?
dense_features/cards/ReshapeReshapedense_features/cards/Cast:y:0+dense_features/cards/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dense_features/cards/Reshape?
dense_features/five_lander/CastCastinputs_five_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2!
dense_features/five_lander/Cast?
 dense_features/five_lander/ShapeShape#dense_features/five_lander/Cast:y:0*
T0*
_output_shapes
:2"
 dense_features/five_lander/Shape?
.dense_features/five_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.dense_features/five_lander/strided_slice/stack?
0dense_features/five_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0dense_features/five_lander/strided_slice/stack_1?
0dense_features/five_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0dense_features/five_lander/strided_slice/stack_2?
(dense_features/five_lander/strided_sliceStridedSlice)dense_features/five_lander/Shape:output:07dense_features/five_lander/strided_slice/stack:output:09dense_features/five_lander/strided_slice/stack_1:output:09dense_features/five_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(dense_features/five_lander/strided_slice?
*dense_features/five_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*dense_features/five_lander/Reshape/shape/1?
(dense_features/five_lander/Reshape/shapePack1dense_features/five_lander/strided_slice:output:03dense_features/five_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2*
(dense_features/five_lander/Reshape/shape?
"dense_features/five_lander/ReshapeReshape#dense_features/five_lander/Cast:y:01dense_features/five_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2$
"dense_features/five_lander/Reshape?
dense_features/four_lander/CastCastinputs_four_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2!
dense_features/four_lander/Cast?
 dense_features/four_lander/ShapeShape#dense_features/four_lander/Cast:y:0*
T0*
_output_shapes
:2"
 dense_features/four_lander/Shape?
.dense_features/four_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.dense_features/four_lander/strided_slice/stack?
0dense_features/four_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0dense_features/four_lander/strided_slice/stack_1?
0dense_features/four_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0dense_features/four_lander/strided_slice/stack_2?
(dense_features/four_lander/strided_sliceStridedSlice)dense_features/four_lander/Shape:output:07dense_features/four_lander/strided_slice/stack:output:09dense_features/four_lander/strided_slice/stack_1:output:09dense_features/four_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(dense_features/four_lander/strided_slice?
*dense_features/four_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*dense_features/four_lander/Reshape/shape/1?
(dense_features/four_lander/Reshape/shapePack1dense_features/four_lander/strided_slice:output:03dense_features/four_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2*
(dense_features/four_lander/Reshape/shape?
"dense_features/four_lander/ReshapeReshape#dense_features/four_lander/Cast:y:01dense_features/four_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2$
"dense_features/four_lander/Reshape?
dense_features/fours/CastCastinputs_fours*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
dense_features/fours/Cast?
dense_features/fours/ShapeShapedense_features/fours/Cast:y:0*
T0*
_output_shapes
:2
dense_features/fours/Shape?
(dense_features/fours/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(dense_features/fours/strided_slice/stack?
*dense_features/fours/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features/fours/strided_slice/stack_1?
*dense_features/fours/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features/fours/strided_slice/stack_2?
"dense_features/fours/strided_sliceStridedSlice#dense_features/fours/Shape:output:01dense_features/fours/strided_slice/stack:output:03dense_features/fours/strided_slice/stack_1:output:03dense_features/fours/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"dense_features/fours/strided_slice?
$dense_features/fours/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$dense_features/fours/Reshape/shape/1?
"dense_features/fours/Reshape/shapePack+dense_features/fours/strided_slice:output:0-dense_features/fours/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2$
"dense_features/fours/Reshape/shape?
dense_features/fours/ReshapeReshapedense_features/fours/Cast:y:0+dense_features/fours/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dense_features/fours/Reshape?
dense_features/on_play/CastCastinputs_on_play*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
dense_features/on_play/Cast?
dense_features/on_play/ShapeShapedense_features/on_play/Cast:y:0*
T0*
_output_shapes
:2
dense_features/on_play/Shape?
*dense_features/on_play/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*dense_features/on_play/strided_slice/stack?
,dense_features/on_play/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features/on_play/strided_slice/stack_1?
,dense_features/on_play/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features/on_play/strided_slice/stack_2?
$dense_features/on_play/strided_sliceStridedSlice%dense_features/on_play/Shape:output:03dense_features/on_play/strided_slice/stack:output:05dense_features/on_play/strided_slice/stack_1:output:05dense_features/on_play/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$dense_features/on_play/strided_slice?
&dense_features/on_play/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&dense_features/on_play/Reshape/shape/1?
$dense_features/on_play/Reshape/shapePack-dense_features/on_play/strided_slice:output:0/dense_features/on_play/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$dense_features/on_play/Reshape/shape?
dense_features/on_play/ReshapeReshapedense_features/on_play/Cast:y:0-dense_features/on_play/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2 
dense_features/on_play/Reshape?
dense_features/ones/CastCastinputs_ones*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
dense_features/ones/Cast?
dense_features/ones/ShapeShapedense_features/ones/Cast:y:0*
T0*
_output_shapes
:2
dense_features/ones/Shape?
'dense_features/ones/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'dense_features/ones/strided_slice/stack?
)dense_features/ones/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)dense_features/ones/strided_slice/stack_1?
)dense_features/ones/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)dense_features/ones/strided_slice/stack_2?
!dense_features/ones/strided_sliceStridedSlice"dense_features/ones/Shape:output:00dense_features/ones/strided_slice/stack:output:02dense_features/ones/strided_slice/stack_1:output:02dense_features/ones/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!dense_features/ones/strided_slice?
#dense_features/ones/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#dense_features/ones/Reshape/shape/1?
!dense_features/ones/Reshape/shapePack*dense_features/ones/strided_slice:output:0,dense_features/ones/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2#
!dense_features/ones/Reshape/shape?
dense_features/ones/ReshapeReshapedense_features/ones/Cast:y:0*dense_features/ones/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dense_features/ones/Reshape?
 dense_features/three_lander/CastCastinputs_three_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2"
 dense_features/three_lander/Cast?
!dense_features/three_lander/ShapeShape$dense_features/three_lander/Cast:y:0*
T0*
_output_shapes
:2#
!dense_features/three_lander/Shape?
/dense_features/three_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_features/three_lander/strided_slice/stack?
1dense_features/three_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_features/three_lander/strided_slice/stack_1?
1dense_features/three_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_features/three_lander/strided_slice/stack_2?
)dense_features/three_lander/strided_sliceStridedSlice*dense_features/three_lander/Shape:output:08dense_features/three_lander/strided_slice/stack:output:0:dense_features/three_lander/strided_slice/stack_1:output:0:dense_features/three_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_features/three_lander/strided_slice?
+dense_features/three_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+dense_features/three_lander/Reshape/shape/1?
)dense_features/three_lander/Reshape/shapePack2dense_features/three_lander/strided_slice:output:04dense_features/three_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2+
)dense_features/three_lander/Reshape/shape?
#dense_features/three_lander/ReshapeReshape$dense_features/three_lander/Cast:y:02dense_features/three_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2%
#dense_features/three_lander/Reshape?
dense_features/threes/CastCastinputs_threes*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
dense_features/threes/Cast?
dense_features/threes/ShapeShapedense_features/threes/Cast:y:0*
T0*
_output_shapes
:2
dense_features/threes/Shape?
)dense_features/threes/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)dense_features/threes/strided_slice/stack?
+dense_features/threes/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features/threes/strided_slice/stack_1?
+dense_features/threes/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features/threes/strided_slice/stack_2?
#dense_features/threes/strided_sliceStridedSlice$dense_features/threes/Shape:output:02dense_features/threes/strided_slice/stack:output:04dense_features/threes/strided_slice/stack_1:output:04dense_features/threes/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#dense_features/threes/strided_slice?
%dense_features/threes/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%dense_features/threes/Reshape/shape/1?
#dense_features/threes/Reshape/shapePack,dense_features/threes/strided_slice:output:0.dense_features/threes/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#dense_features/threes/Reshape/shape?
dense_features/threes/ReshapeReshapedense_features/threes/Cast:y:0,dense_features/threes/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dense_features/threes/Reshape?
dense_features/two_lander/CastCastinputs_two_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2 
dense_features/two_lander/Cast?
dense_features/two_lander/ShapeShape"dense_features/two_lander/Cast:y:0*
T0*
_output_shapes
:2!
dense_features/two_lander/Shape?
-dense_features/two_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense_features/two_lander/strided_slice/stack?
/dense_features/two_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense_features/two_lander/strided_slice/stack_1?
/dense_features/two_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense_features/two_lander/strided_slice/stack_2?
'dense_features/two_lander/strided_sliceStridedSlice(dense_features/two_lander/Shape:output:06dense_features/two_lander/strided_slice/stack:output:08dense_features/two_lander/strided_slice/stack_1:output:08dense_features/two_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense_features/two_lander/strided_slice?
)dense_features/two_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)dense_features/two_lander/Reshape/shape/1?
'dense_features/two_lander/Reshape/shapePack0dense_features/two_lander/strided_slice:output:02dense_features/two_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2)
'dense_features/two_lander/Reshape/shape?
!dense_features/two_lander/ReshapeReshape"dense_features/two_lander/Cast:y:00dense_features/two_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2#
!dense_features/two_lander/Reshape?
dense_features/twos/CastCastinputs_twos*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
dense_features/twos/Cast?
dense_features/twos/ShapeShapedense_features/twos/Cast:y:0*
T0*
_output_shapes
:2
dense_features/twos/Shape?
'dense_features/twos/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'dense_features/twos/strided_slice/stack?
)dense_features/twos/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)dense_features/twos/strided_slice/stack_1?
)dense_features/twos/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)dense_features/twos/strided_slice/stack_2?
!dense_features/twos/strided_sliceStridedSlice"dense_features/twos/Shape:output:00dense_features/twos/strided_slice/stack:output:02dense_features/twos/strided_slice/stack_1:output:02dense_features/twos/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!dense_features/twos/strided_slice?
#dense_features/twos/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#dense_features/twos/Reshape/shape/1?
!dense_features/twos/Reshape/shapePack*dense_features/twos/strided_slice:output:0,dense_features/twos/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2#
!dense_features/twos/Reshape/shape?
dense_features/twos/ReshapeReshapedense_features/twos/Cast:y:0*dense_features/twos/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dense_features/twos/Reshape?
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
dense_features/concat/axis?
dense_features/concatConcatV2%dense_features/cards/Reshape:output:0+dense_features/five_lander/Reshape:output:0+dense_features/four_lander/Reshape:output:0%dense_features/fours/Reshape:output:0'dense_features/on_play/Reshape:output:0$dense_features/ones/Reshape:output:0,dense_features/three_lander/Reshape:output:0&dense_features/threes/Reshape:output:0*dense_features/two_lander/Reshape:output:0$dense_features/twos/Reshape:output:0#dense_features/concat/axis:output:0*
N
*
T0*'
_output_shapes
:?????????
2
dense_features/concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMuldense_features/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense/Sigmoidl
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:U Q
'
_output_shapes
:?????????
&
_user_specified_nameinputs/cards:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/five_lander:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/four_lander:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/fours:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/on_play:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/ones:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/three_lander:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/threes:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/two_lander:T	P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/twos
?
?
-__inference_sequential_layer_call_fn_11236597
inputs_cards	
inputs_five_lander	
inputs_four_lander	
inputs_fours	
inputs_on_play	
inputs_ones	
inputs_three_lander	
inputs_threes	
inputs_two_lander	
inputs_twos	
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_cardsinputs_five_landerinputs_four_landerinputs_foursinputs_on_playinputs_onesinputs_three_landerinputs_threesinputs_two_landerinputs_twosunknown	unknown_0*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_112360672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_nameinputs/cards:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/five_lander:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/four_lander:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/fours:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/on_play:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/ones:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/three_lander:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/threes:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/two_lander:T	P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/twos
?
?
1__inference_dense_features_layer_call_fn_11236839
features_cards	
features_five_lander	
features_four_lander	
features_fours	
features_on_play	
features_ones	
features_three_lander	
features_threes	
features_two_lander	
features_twos	
identity?
PartitionedCallPartitionedCallfeatures_cardsfeatures_five_landerfeatures_four_landerfeatures_foursfeatures_on_playfeatures_onesfeatures_three_landerfeatures_threesfeatures_two_landerfeatures_twos*
Tin
2
										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_dense_features_layer_call_and_return_conditional_losses_112360472
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:W S
'
_output_shapes
:?????????
(
_user_specified_namefeatures/cards:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/five_lander:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/four_lander:WS
'
_output_shapes
:?????????
(
_user_specified_namefeatures/fours:YU
'
_output_shapes
:?????????
*
_user_specified_namefeatures/on_play:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/ones:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/three_lander:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/threes:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/two_lander:V	R
'
_output_shapes
:?????????
'
_user_specified_namefeatures/twos
?
?
-__inference_sequential_layer_call_fn_11236074	
cards	
five_lander	
four_lander		
fours	
on_play	
ones	
three_lander	

threes	

two_lander	
twos	
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcardsfive_landerfour_landerfourson_playonesthree_landerthrees
two_landertwosunknown	unknown_0*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_112360672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namecards:TP
'
_output_shapes
:?????????
%
_user_specified_namefive_lander:TP
'
_output_shapes
:?????????
%
_user_specified_namefour_lander:NJ
'
_output_shapes
:?????????

_user_specified_namefours:PL
'
_output_shapes
:?????????
!
_user_specified_name	on_play:MI
'
_output_shapes
:?????????

_user_specified_nameones:UQ
'
_output_shapes
:?????????
&
_user_specified_namethree_lander:OK
'
_output_shapes
:?????????
 
_user_specified_namethrees:SO
'
_output_shapes
:?????????
$
_user_specified_name
two_lander:M	I
'
_output_shapes
:?????????

_user_specified_nametwos
??
?
#__inference__wrapped_model_11235917	
cards	
five_lander	
four_lander		
fours	
on_play	
ones	
three_lander	

threes	

two_lander	
twos	A
/sequential_dense_matmul_readvariableop_resource:
>
0sequential_dense_biasadd_readvariableop_resource:
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
$sequential/dense_features/cards/CastCastcards*

DstT0*

SrcT0	*'
_output_shapes
:?????????2&
$sequential/dense_features/cards/Cast?
%sequential/dense_features/cards/ShapeShape(sequential/dense_features/cards/Cast:y:0*
T0*
_output_shapes
:2'
%sequential/dense_features/cards/Shape?
3sequential/dense_features/cards/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential/dense_features/cards/strided_slice/stack?
5sequential/dense_features/cards/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/dense_features/cards/strided_slice/stack_1?
5sequential/dense_features/cards/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/dense_features/cards/strided_slice/stack_2?
-sequential/dense_features/cards/strided_sliceStridedSlice.sequential/dense_features/cards/Shape:output:0<sequential/dense_features/cards/strided_slice/stack:output:0>sequential/dense_features/cards/strided_slice/stack_1:output:0>sequential/dense_features/cards/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/dense_features/cards/strided_slice?
/sequential/dense_features/cards/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/sequential/dense_features/cards/Reshape/shape/1?
-sequential/dense_features/cards/Reshape/shapePack6sequential/dense_features/cards/strided_slice:output:08sequential/dense_features/cards/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-sequential/dense_features/cards/Reshape/shape?
'sequential/dense_features/cards/ReshapeReshape(sequential/dense_features/cards/Cast:y:06sequential/dense_features/cards/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2)
'sequential/dense_features/cards/Reshape?
*sequential/dense_features/five_lander/CastCastfive_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2,
*sequential/dense_features/five_lander/Cast?
+sequential/dense_features/five_lander/ShapeShape.sequential/dense_features/five_lander/Cast:y:0*
T0*
_output_shapes
:2-
+sequential/dense_features/five_lander/Shape?
9sequential/dense_features/five_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9sequential/dense_features/five_lander/strided_slice/stack?
;sequential/dense_features/five_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;sequential/dense_features/five_lander/strided_slice/stack_1?
;sequential/dense_features/five_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;sequential/dense_features/five_lander/strided_slice/stack_2?
3sequential/dense_features/five_lander/strided_sliceStridedSlice4sequential/dense_features/five_lander/Shape:output:0Bsequential/dense_features/five_lander/strided_slice/stack:output:0Dsequential/dense_features/five_lander/strided_slice/stack_1:output:0Dsequential/dense_features/five_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3sequential/dense_features/five_lander/strided_slice?
5sequential/dense_features/five_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5sequential/dense_features/five_lander/Reshape/shape/1?
3sequential/dense_features/five_lander/Reshape/shapePack<sequential/dense_features/five_lander/strided_slice:output:0>sequential/dense_features/five_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:25
3sequential/dense_features/five_lander/Reshape/shape?
-sequential/dense_features/five_lander/ReshapeReshape.sequential/dense_features/five_lander/Cast:y:0<sequential/dense_features/five_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2/
-sequential/dense_features/five_lander/Reshape?
*sequential/dense_features/four_lander/CastCastfour_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2,
*sequential/dense_features/four_lander/Cast?
+sequential/dense_features/four_lander/ShapeShape.sequential/dense_features/four_lander/Cast:y:0*
T0*
_output_shapes
:2-
+sequential/dense_features/four_lander/Shape?
9sequential/dense_features/four_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9sequential/dense_features/four_lander/strided_slice/stack?
;sequential/dense_features/four_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;sequential/dense_features/four_lander/strided_slice/stack_1?
;sequential/dense_features/four_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;sequential/dense_features/four_lander/strided_slice/stack_2?
3sequential/dense_features/four_lander/strided_sliceStridedSlice4sequential/dense_features/four_lander/Shape:output:0Bsequential/dense_features/four_lander/strided_slice/stack:output:0Dsequential/dense_features/four_lander/strided_slice/stack_1:output:0Dsequential/dense_features/four_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3sequential/dense_features/four_lander/strided_slice?
5sequential/dense_features/four_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5sequential/dense_features/four_lander/Reshape/shape/1?
3sequential/dense_features/four_lander/Reshape/shapePack<sequential/dense_features/four_lander/strided_slice:output:0>sequential/dense_features/four_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:25
3sequential/dense_features/four_lander/Reshape/shape?
-sequential/dense_features/four_lander/ReshapeReshape.sequential/dense_features/four_lander/Cast:y:0<sequential/dense_features/four_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2/
-sequential/dense_features/four_lander/Reshape?
$sequential/dense_features/fours/CastCastfours*

DstT0*

SrcT0	*'
_output_shapes
:?????????2&
$sequential/dense_features/fours/Cast?
%sequential/dense_features/fours/ShapeShape(sequential/dense_features/fours/Cast:y:0*
T0*
_output_shapes
:2'
%sequential/dense_features/fours/Shape?
3sequential/dense_features/fours/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential/dense_features/fours/strided_slice/stack?
5sequential/dense_features/fours/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/dense_features/fours/strided_slice/stack_1?
5sequential/dense_features/fours/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/dense_features/fours/strided_slice/stack_2?
-sequential/dense_features/fours/strided_sliceStridedSlice.sequential/dense_features/fours/Shape:output:0<sequential/dense_features/fours/strided_slice/stack:output:0>sequential/dense_features/fours/strided_slice/stack_1:output:0>sequential/dense_features/fours/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/dense_features/fours/strided_slice?
/sequential/dense_features/fours/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/sequential/dense_features/fours/Reshape/shape/1?
-sequential/dense_features/fours/Reshape/shapePack6sequential/dense_features/fours/strided_slice:output:08sequential/dense_features/fours/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-sequential/dense_features/fours/Reshape/shape?
'sequential/dense_features/fours/ReshapeReshape(sequential/dense_features/fours/Cast:y:06sequential/dense_features/fours/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2)
'sequential/dense_features/fours/Reshape?
&sequential/dense_features/on_play/CastCaston_play*

DstT0*

SrcT0	*'
_output_shapes
:?????????2(
&sequential/dense_features/on_play/Cast?
'sequential/dense_features/on_play/ShapeShape*sequential/dense_features/on_play/Cast:y:0*
T0*
_output_shapes
:2)
'sequential/dense_features/on_play/Shape?
5sequential/dense_features/on_play/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential/dense_features/on_play/strided_slice/stack?
7sequential/dense_features/on_play/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential/dense_features/on_play/strided_slice/stack_1?
7sequential/dense_features/on_play/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential/dense_features/on_play/strided_slice/stack_2?
/sequential/dense_features/on_play/strided_sliceStridedSlice0sequential/dense_features/on_play/Shape:output:0>sequential/dense_features/on_play/strided_slice/stack:output:0@sequential/dense_features/on_play/strided_slice/stack_1:output:0@sequential/dense_features/on_play/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential/dense_features/on_play/strided_slice?
1sequential/dense_features/on_play/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1sequential/dense_features/on_play/Reshape/shape/1?
/sequential/dense_features/on_play/Reshape/shapePack8sequential/dense_features/on_play/strided_slice:output:0:sequential/dense_features/on_play/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/sequential/dense_features/on_play/Reshape/shape?
)sequential/dense_features/on_play/ReshapeReshape*sequential/dense_features/on_play/Cast:y:08sequential/dense_features/on_play/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2+
)sequential/dense_features/on_play/Reshape?
#sequential/dense_features/ones/CastCastones*

DstT0*

SrcT0	*'
_output_shapes
:?????????2%
#sequential/dense_features/ones/Cast?
$sequential/dense_features/ones/ShapeShape'sequential/dense_features/ones/Cast:y:0*
T0*
_output_shapes
:2&
$sequential/dense_features/ones/Shape?
2sequential/dense_features/ones/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential/dense_features/ones/strided_slice/stack?
4sequential/dense_features/ones/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential/dense_features/ones/strided_slice/stack_1?
4sequential/dense_features/ones/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential/dense_features/ones/strided_slice/stack_2?
,sequential/dense_features/ones/strided_sliceStridedSlice-sequential/dense_features/ones/Shape:output:0;sequential/dense_features/ones/strided_slice/stack:output:0=sequential/dense_features/ones/strided_slice/stack_1:output:0=sequential/dense_features/ones/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential/dense_features/ones/strided_slice?
.sequential/dense_features/ones/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.sequential/dense_features/ones/Reshape/shape/1?
,sequential/dense_features/ones/Reshape/shapePack5sequential/dense_features/ones/strided_slice:output:07sequential/dense_features/ones/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,sequential/dense_features/ones/Reshape/shape?
&sequential/dense_features/ones/ReshapeReshape'sequential/dense_features/ones/Cast:y:05sequential/dense_features/ones/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2(
&sequential/dense_features/ones/Reshape?
+sequential/dense_features/three_lander/CastCastthree_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2-
+sequential/dense_features/three_lander/Cast?
,sequential/dense_features/three_lander/ShapeShape/sequential/dense_features/three_lander/Cast:y:0*
T0*
_output_shapes
:2.
,sequential/dense_features/three_lander/Shape?
:sequential/dense_features/three_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:sequential/dense_features/three_lander/strided_slice/stack?
<sequential/dense_features/three_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential/dense_features/three_lander/strided_slice/stack_1?
<sequential/dense_features/three_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential/dense_features/three_lander/strided_slice/stack_2?
4sequential/dense_features/three_lander/strided_sliceStridedSlice5sequential/dense_features/three_lander/Shape:output:0Csequential/dense_features/three_lander/strided_slice/stack:output:0Esequential/dense_features/three_lander/strided_slice/stack_1:output:0Esequential/dense_features/three_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4sequential/dense_features/three_lander/strided_slice?
6sequential/dense_features/three_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :28
6sequential/dense_features/three_lander/Reshape/shape/1?
4sequential/dense_features/three_lander/Reshape/shapePack=sequential/dense_features/three_lander/strided_slice:output:0?sequential/dense_features/three_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:26
4sequential/dense_features/three_lander/Reshape/shape?
.sequential/dense_features/three_lander/ReshapeReshape/sequential/dense_features/three_lander/Cast:y:0=sequential/dense_features/three_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????20
.sequential/dense_features/three_lander/Reshape?
%sequential/dense_features/threes/CastCastthrees*

DstT0*

SrcT0	*'
_output_shapes
:?????????2'
%sequential/dense_features/threes/Cast?
&sequential/dense_features/threes/ShapeShape)sequential/dense_features/threes/Cast:y:0*
T0*
_output_shapes
:2(
&sequential/dense_features/threes/Shape?
4sequential/dense_features/threes/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential/dense_features/threes/strided_slice/stack?
6sequential/dense_features/threes/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential/dense_features/threes/strided_slice/stack_1?
6sequential/dense_features/threes/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential/dense_features/threes/strided_slice/stack_2?
.sequential/dense_features/threes/strided_sliceStridedSlice/sequential/dense_features/threes/Shape:output:0=sequential/dense_features/threes/strided_slice/stack:output:0?sequential/dense_features/threes/strided_slice/stack_1:output:0?sequential/dense_features/threes/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential/dense_features/threes/strided_slice?
0sequential/dense_features/threes/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0sequential/dense_features/threes/Reshape/shape/1?
.sequential/dense_features/threes/Reshape/shapePack7sequential/dense_features/threes/strided_slice:output:09sequential/dense_features/threes/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.sequential/dense_features/threes/Reshape/shape?
(sequential/dense_features/threes/ReshapeReshape)sequential/dense_features/threes/Cast:y:07sequential/dense_features/threes/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2*
(sequential/dense_features/threes/Reshape?
)sequential/dense_features/two_lander/CastCast
two_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2+
)sequential/dense_features/two_lander/Cast?
*sequential/dense_features/two_lander/ShapeShape-sequential/dense_features/two_lander/Cast:y:0*
T0*
_output_shapes
:2,
*sequential/dense_features/two_lander/Shape?
8sequential/dense_features/two_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential/dense_features/two_lander/strided_slice/stack?
:sequential/dense_features/two_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential/dense_features/two_lander/strided_slice/stack_1?
:sequential/dense_features/two_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential/dense_features/two_lander/strided_slice/stack_2?
2sequential/dense_features/two_lander/strided_sliceStridedSlice3sequential/dense_features/two_lander/Shape:output:0Asequential/dense_features/two_lander/strided_slice/stack:output:0Csequential/dense_features/two_lander/strided_slice/stack_1:output:0Csequential/dense_features/two_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2sequential/dense_features/two_lander/strided_slice?
4sequential/dense_features/two_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :26
4sequential/dense_features/two_lander/Reshape/shape/1?
2sequential/dense_features/two_lander/Reshape/shapePack;sequential/dense_features/two_lander/strided_slice:output:0=sequential/dense_features/two_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:24
2sequential/dense_features/two_lander/Reshape/shape?
,sequential/dense_features/two_lander/ReshapeReshape-sequential/dense_features/two_lander/Cast:y:0;sequential/dense_features/two_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2.
,sequential/dense_features/two_lander/Reshape?
#sequential/dense_features/twos/CastCasttwos*

DstT0*

SrcT0	*'
_output_shapes
:?????????2%
#sequential/dense_features/twos/Cast?
$sequential/dense_features/twos/ShapeShape'sequential/dense_features/twos/Cast:y:0*
T0*
_output_shapes
:2&
$sequential/dense_features/twos/Shape?
2sequential/dense_features/twos/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential/dense_features/twos/strided_slice/stack?
4sequential/dense_features/twos/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential/dense_features/twos/strided_slice/stack_1?
4sequential/dense_features/twos/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential/dense_features/twos/strided_slice/stack_2?
,sequential/dense_features/twos/strided_sliceStridedSlice-sequential/dense_features/twos/Shape:output:0;sequential/dense_features/twos/strided_slice/stack:output:0=sequential/dense_features/twos/strided_slice/stack_1:output:0=sequential/dense_features/twos/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential/dense_features/twos/strided_slice?
.sequential/dense_features/twos/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.sequential/dense_features/twos/Reshape/shape/1?
,sequential/dense_features/twos/Reshape/shapePack5sequential/dense_features/twos/strided_slice:output:07sequential/dense_features/twos/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,sequential/dense_features/twos/Reshape/shape?
&sequential/dense_features/twos/ReshapeReshape'sequential/dense_features/twos/Cast:y:05sequential/dense_features/twos/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2(
&sequential/dense_features/twos/Reshape?
%sequential/dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%sequential/dense_features/concat/axis?
 sequential/dense_features/concatConcatV20sequential/dense_features/cards/Reshape:output:06sequential/dense_features/five_lander/Reshape:output:06sequential/dense_features/four_lander/Reshape:output:00sequential/dense_features/fours/Reshape:output:02sequential/dense_features/on_play/Reshape:output:0/sequential/dense_features/ones/Reshape:output:07sequential/dense_features/three_lander/Reshape:output:01sequential/dense_features/threes/Reshape:output:05sequential/dense_features/two_lander/Reshape:output:0/sequential/dense_features/twos/Reshape:output:0.sequential/dense_features/concat/axis:output:0*
N
*
T0*'
_output_shapes
:?????????
2"
 sequential/dense_features/concat?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul)sequential/dense_features/concat:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/BiasAdd?
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense/Sigmoidw
IdentityIdentitysequential/dense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_namecards:TP
'
_output_shapes
:?????????
%
_user_specified_namefive_lander:TP
'
_output_shapes
:?????????
%
_user_specified_namefour_lander:NJ
'
_output_shapes
:?????????

_user_specified_namefours:PL
'
_output_shapes
:?????????
!
_user_specified_name	on_play:MI
'
_output_shapes
:?????????

_user_specified_nameones:UQ
'
_output_shapes
:?????????
&
_user_specified_namethree_lander:OK
'
_output_shapes
:?????????
 
_user_specified_namethrees:SO
'
_output_shapes
:?????????
$
_user_specified_name
two_lander:M	I
'
_output_shapes
:?????????

_user_specified_nametwos
?B
?
$__inference__traced_restore_11237005
file_prefix:
(assignvariableop_sequential_dense_kernel:
6
(assignvariableop_1_sequential_dense_bias:)
assignvariableop_2_rmsprop_iter:	 *
 assignvariableop_3_rmsprop_decay: 2
(assignvariableop_4_rmsprop_learning_rate: -
#assignvariableop_5_rmsprop_momentum: (
assignvariableop_6_rmsprop_rho: "
assignvariableop_7_total: "
assignvariableop_8_count: $
assignvariableop_9_total_1: %
assignvariableop_10_count_1: %
assignvariableop_11_total_2: %
assignvariableop_12_count_2: I
7assignvariableop_13_rmsprop_sequential_dense_kernel_rms:
C
5assignvariableop_14_rmsprop_sequential_dense_bias_rms:
identity_16??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp(assignvariableop_sequential_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp(assignvariableop_1_sequential_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_rmsprop_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_rmsprop_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp(assignvariableop_4_rmsprop_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_rmsprop_momentumIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_rhoIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_total_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_count_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp7assignvariableop_13_rmsprop_sequential_dense_kernel_rmsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp5assignvariableop_14_rmsprop_sequential_dense_bias_rmsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_149
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_15Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_15f
Identity_16IdentityIdentity_15:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_16?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_16Identity_16:output:0*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?z
?
L__inference_dense_features_layer_call_and_return_conditional_losses_11236825
features_cards	
features_five_lander	
features_four_lander	
features_fours	
features_on_play	
features_ones	
features_three_lander	
features_threes	
features_two_lander	
features_twos	
identityq

cards/CastCastfeatures_cards*

DstT0*

SrcT0	*'
_output_shapes
:?????????2

cards/CastX
cards/ShapeShapecards/Cast:y:0*
T0*
_output_shapes
:2
cards/Shape?
cards/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cards/strided_slice/stack?
cards/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cards/strided_slice/stack_1?
cards/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cards/strided_slice/stack_2?
cards/strided_sliceStridedSlicecards/Shape:output:0"cards/strided_slice/stack:output:0$cards/strided_slice/stack_1:output:0$cards/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cards/strided_slicep
cards/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
cards/Reshape/shape/1?
cards/Reshape/shapePackcards/strided_slice:output:0cards/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
cards/Reshape/shape?
cards/ReshapeReshapecards/Cast:y:0cards/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
cards/Reshape?
five_lander/CastCastfeatures_five_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
five_lander/Castj
five_lander/ShapeShapefive_lander/Cast:y:0*
T0*
_output_shapes
:2
five_lander/Shape?
five_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
five_lander/strided_slice/stack?
!five_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!five_lander/strided_slice/stack_1?
!five_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!five_lander/strided_slice/stack_2?
five_lander/strided_sliceStridedSlicefive_lander/Shape:output:0(five_lander/strided_slice/stack:output:0*five_lander/strided_slice/stack_1:output:0*five_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
five_lander/strided_slice|
five_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
five_lander/Reshape/shape/1?
five_lander/Reshape/shapePack"five_lander/strided_slice:output:0$five_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
five_lander/Reshape/shape?
five_lander/ReshapeReshapefive_lander/Cast:y:0"five_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
five_lander/Reshape?
four_lander/CastCastfeatures_four_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
four_lander/Castj
four_lander/ShapeShapefour_lander/Cast:y:0*
T0*
_output_shapes
:2
four_lander/Shape?
four_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
four_lander/strided_slice/stack?
!four_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!four_lander/strided_slice/stack_1?
!four_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!four_lander/strided_slice/stack_2?
four_lander/strided_sliceStridedSlicefour_lander/Shape:output:0(four_lander/strided_slice/stack:output:0*four_lander/strided_slice/stack_1:output:0*four_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
four_lander/strided_slice|
four_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
four_lander/Reshape/shape/1?
four_lander/Reshape/shapePack"four_lander/strided_slice:output:0$four_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
four_lander/Reshape/shape?
four_lander/ReshapeReshapefour_lander/Cast:y:0"four_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
four_lander/Reshapeq

fours/CastCastfeatures_fours*

DstT0*

SrcT0	*'
_output_shapes
:?????????2

fours/CastX
fours/ShapeShapefours/Cast:y:0*
T0*
_output_shapes
:2
fours/Shape?
fours/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
fours/strided_slice/stack?
fours/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
fours/strided_slice/stack_1?
fours/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
fours/strided_slice/stack_2?
fours/strided_sliceStridedSlicefours/Shape:output:0"fours/strided_slice/stack:output:0$fours/strided_slice/stack_1:output:0$fours/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
fours/strided_slicep
fours/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
fours/Reshape/shape/1?
fours/Reshape/shapePackfours/strided_slice:output:0fours/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
fours/Reshape/shape?
fours/ReshapeReshapefours/Cast:y:0fours/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
fours/Reshapew
on_play/CastCastfeatures_on_play*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
on_play/Cast^
on_play/ShapeShapeon_play/Cast:y:0*
T0*
_output_shapes
:2
on_play/Shape?
on_play/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
on_play/strided_slice/stack?
on_play/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
on_play/strided_slice/stack_1?
on_play/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
on_play/strided_slice/stack_2?
on_play/strided_sliceStridedSliceon_play/Shape:output:0$on_play/strided_slice/stack:output:0&on_play/strided_slice/stack_1:output:0&on_play/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
on_play/strided_slicet
on_play/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
on_play/Reshape/shape/1?
on_play/Reshape/shapePackon_play/strided_slice:output:0 on_play/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
on_play/Reshape/shape?
on_play/ReshapeReshapeon_play/Cast:y:0on_play/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
on_play/Reshapen
	ones/CastCastfeatures_ones*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
	ones/CastU

ones/ShapeShapeones/Cast:y:0*
T0*
_output_shapes
:2

ones/Shape~
ones/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
ones/strided_slice/stack?
ones/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
ones/strided_slice/stack_1?
ones/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
ones/strided_slice/stack_2?
ones/strided_sliceStridedSliceones/Shape:output:0!ones/strided_slice/stack:output:0#ones/strided_slice/stack_1:output:0#ones/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ones/strided_slicen
ones/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
ones/Reshape/shape/1?
ones/Reshape/shapePackones/strided_slice:output:0ones/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
ones/Reshape/shape?
ones/ReshapeReshapeones/Cast:y:0ones/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
ones/Reshape?
three_lander/CastCastfeatures_three_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
three_lander/Castm
three_lander/ShapeShapethree_lander/Cast:y:0*
T0*
_output_shapes
:2
three_lander/Shape?
 three_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 three_lander/strided_slice/stack?
"three_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"three_lander/strided_slice/stack_1?
"three_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"three_lander/strided_slice/stack_2?
three_lander/strided_sliceStridedSlicethree_lander/Shape:output:0)three_lander/strided_slice/stack:output:0+three_lander/strided_slice/stack_1:output:0+three_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
three_lander/strided_slice~
three_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
three_lander/Reshape/shape/1?
three_lander/Reshape/shapePack#three_lander/strided_slice:output:0%three_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
three_lander/Reshape/shape?
three_lander/ReshapeReshapethree_lander/Cast:y:0#three_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
three_lander/Reshapet
threes/CastCastfeatures_threes*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
threes/Cast[
threes/ShapeShapethrees/Cast:y:0*
T0*
_output_shapes
:2
threes/Shape?
threes/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
threes/strided_slice/stack?
threes/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
threes/strided_slice/stack_1?
threes/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
threes/strided_slice/stack_2?
threes/strided_sliceStridedSlicethrees/Shape:output:0#threes/strided_slice/stack:output:0%threes/strided_slice/stack_1:output:0%threes/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
threes/strided_slicer
threes/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
threes/Reshape/shape/1?
threes/Reshape/shapePackthrees/strided_slice:output:0threes/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
threes/Reshape/shape?
threes/ReshapeReshapethrees/Cast:y:0threes/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
threes/Reshape?
two_lander/CastCastfeatures_two_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
two_lander/Castg
two_lander/ShapeShapetwo_lander/Cast:y:0*
T0*
_output_shapes
:2
two_lander/Shape?
two_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
two_lander/strided_slice/stack?
 two_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 two_lander/strided_slice/stack_1?
 two_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 two_lander/strided_slice/stack_2?
two_lander/strided_sliceStridedSlicetwo_lander/Shape:output:0'two_lander/strided_slice/stack:output:0)two_lander/strided_slice/stack_1:output:0)two_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
two_lander/strided_slicez
two_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
two_lander/Reshape/shape/1?
two_lander/Reshape/shapePack!two_lander/strided_slice:output:0#two_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
two_lander/Reshape/shape?
two_lander/ReshapeReshapetwo_lander/Cast:y:0!two_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
two_lander/Reshapen
	twos/CastCastfeatures_twos*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
	twos/CastU

twos/ShapeShapetwos/Cast:y:0*
T0*
_output_shapes
:2

twos/Shape~
twos/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
twos/strided_slice/stack?
twos/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
twos/strided_slice/stack_1?
twos/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
twos/strided_slice/stack_2?
twos/strided_sliceStridedSlicetwos/Shape:output:0!twos/strided_slice/stack:output:0#twos/strided_slice/stack_1:output:0#twos/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
twos/strided_slicen
twos/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
twos/Reshape/shape/1?
twos/Reshape/shapePacktwos/strided_slice:output:0twos/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
twos/Reshape/shape?
twos/ReshapeReshapetwos/Cast:y:0twos/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
twos/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2cards/Reshape:output:0five_lander/Reshape:output:0four_lander/Reshape:output:0fours/Reshape:output:0on_play/Reshape:output:0ones/Reshape:output:0three_lander/Reshape:output:0threes/Reshape:output:0two_lander/Reshape:output:0twos/Reshape:output:0concat/axis:output:0*
N
*
T0*'
_output_shapes
:?????????
2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:W S
'
_output_shapes
:?????????
(
_user_specified_namefeatures/cards:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/five_lander:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/four_lander:WS
'
_output_shapes
:?????????
(
_user_specified_namefeatures/fours:YU
'
_output_shapes
:?????????
*
_user_specified_namefeatures/on_play:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/ones:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/three_lander:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/threes:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/two_lander:V	R
'
_output_shapes
:?????????
'
_user_specified_namefeatures/twos
?
?
1__inference_dense_features_layer_call_fn_11236853
features_cards	
features_five_lander	
features_four_lander	
features_fours	
features_on_play	
features_ones	
features_three_lander	
features_threes	
features_two_lander	
features_twos	
identity?
PartitionedCallPartitionedCallfeatures_cardsfeatures_five_landerfeatures_four_landerfeatures_foursfeatures_on_playfeatures_onesfeatures_three_landerfeatures_threesfeatures_two_landerfeatures_twos*
Tin
2
										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_dense_features_layer_call_and_return_conditional_losses_112362152
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:W S
'
_output_shapes
:?????????
(
_user_specified_namefeatures/cards:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/five_lander:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/four_lander:WS
'
_output_shapes
:?????????
(
_user_specified_namefeatures/fours:YU
'
_output_shapes
:?????????
*
_user_specified_namefeatures/on_play:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/ones:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/three_lander:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/threes:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/two_lander:V	R
'
_output_shapes
:?????????
'
_user_specified_namefeatures/twos
?
?
H__inference_sequential_layer_call_and_return_conditional_losses_11236329	
cards	
five_lander	
four_lander		
fours	
on_play	
ones	
three_lander	

threes	

two_lander	
twos	 
dense_11236323:

dense_11236325:
identity??dense/StatefulPartitionedCall?
dense_features/PartitionedCallPartitionedCallcardsfive_landerfour_landerfourson_playonesthree_landerthrees
two_landertwos*
Tin
2
										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_dense_features_layer_call_and_return_conditional_losses_112362152 
dense_features/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0dense_11236323dense_11236325*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_112360602
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityn
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namecards:TP
'
_output_shapes
:?????????
%
_user_specified_namefive_lander:TP
'
_output_shapes
:?????????
%
_user_specified_namefour_lander:NJ
'
_output_shapes
:?????????

_user_specified_namefours:PL
'
_output_shapes
:?????????
!
_user_specified_name	on_play:MI
'
_output_shapes
:?????????

_user_specified_nameones:UQ
'
_output_shapes
:?????????
&
_user_specified_namethree_lander:OK
'
_output_shapes
:?????????
 
_user_specified_namethrees:SO
'
_output_shapes
:?????????
$
_user_specified_name
two_lander:M	I
'
_output_shapes
:?????????

_user_specified_nametwos
?
?
H__inference_sequential_layer_call_and_return_conditional_losses_11236310	
cards	
five_lander	
four_lander		
fours	
on_play	
ones	
three_lander	

threes	

two_lander	
twos	 
dense_11236304:

dense_11236306:
identity??dense/StatefulPartitionedCall?
dense_features/PartitionedCallPartitionedCallcardsfive_landerfour_landerfourson_playonesthree_landerthrees
two_landertwos*
Tin
2
										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_dense_features_layer_call_and_return_conditional_losses_112360472 
dense_features/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0dense_11236304dense_11236306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_112360602
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityn
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namecards:TP
'
_output_shapes
:?????????
%
_user_specified_namefive_lander:TP
'
_output_shapes
:?????????
%
_user_specified_namefour_lander:NJ
'
_output_shapes
:?????????

_user_specified_namefours:PL
'
_output_shapes
:?????????
!
_user_specified_name	on_play:MI
'
_output_shapes
:?????????

_user_specified_nameones:UQ
'
_output_shapes
:?????????
&
_user_specified_namethree_lander:OK
'
_output_shapes
:?????????
 
_user_specified_namethrees:SO
'
_output_shapes
:?????????
$
_user_specified_name
two_lander:M	I
'
_output_shapes
:?????????

_user_specified_nametwos
?x
?
L__inference_dense_features_layer_call_and_return_conditional_losses_11236215
features	

features_1	

features_2	

features_3	

features_4	

features_5	

features_6	

features_7	

features_8	

features_9	
identityk

cards/CastCastfeatures*

DstT0*

SrcT0	*'
_output_shapes
:?????????2

cards/CastX
cards/ShapeShapecards/Cast:y:0*
T0*
_output_shapes
:2
cards/Shape?
cards/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cards/strided_slice/stack?
cards/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cards/strided_slice/stack_1?
cards/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cards/strided_slice/stack_2?
cards/strided_sliceStridedSlicecards/Shape:output:0"cards/strided_slice/stack:output:0$cards/strided_slice/stack_1:output:0$cards/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cards/strided_slicep
cards/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
cards/Reshape/shape/1?
cards/Reshape/shapePackcards/strided_slice:output:0cards/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
cards/Reshape/shape?
cards/ReshapeReshapecards/Cast:y:0cards/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
cards/Reshapey
five_lander/CastCast
features_1*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
five_lander/Castj
five_lander/ShapeShapefive_lander/Cast:y:0*
T0*
_output_shapes
:2
five_lander/Shape?
five_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
five_lander/strided_slice/stack?
!five_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!five_lander/strided_slice/stack_1?
!five_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!five_lander/strided_slice/stack_2?
five_lander/strided_sliceStridedSlicefive_lander/Shape:output:0(five_lander/strided_slice/stack:output:0*five_lander/strided_slice/stack_1:output:0*five_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
five_lander/strided_slice|
five_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
five_lander/Reshape/shape/1?
five_lander/Reshape/shapePack"five_lander/strided_slice:output:0$five_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
five_lander/Reshape/shape?
five_lander/ReshapeReshapefive_lander/Cast:y:0"five_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
five_lander/Reshapey
four_lander/CastCast
features_2*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
four_lander/Castj
four_lander/ShapeShapefour_lander/Cast:y:0*
T0*
_output_shapes
:2
four_lander/Shape?
four_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
four_lander/strided_slice/stack?
!four_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!four_lander/strided_slice/stack_1?
!four_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!four_lander/strided_slice/stack_2?
four_lander/strided_sliceStridedSlicefour_lander/Shape:output:0(four_lander/strided_slice/stack:output:0*four_lander/strided_slice/stack_1:output:0*four_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
four_lander/strided_slice|
four_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
four_lander/Reshape/shape/1?
four_lander/Reshape/shapePack"four_lander/strided_slice:output:0$four_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
four_lander/Reshape/shape?
four_lander/ReshapeReshapefour_lander/Cast:y:0"four_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
four_lander/Reshapem

fours/CastCast
features_3*

DstT0*

SrcT0	*'
_output_shapes
:?????????2

fours/CastX
fours/ShapeShapefours/Cast:y:0*
T0*
_output_shapes
:2
fours/Shape?
fours/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
fours/strided_slice/stack?
fours/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
fours/strided_slice/stack_1?
fours/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
fours/strided_slice/stack_2?
fours/strided_sliceStridedSlicefours/Shape:output:0"fours/strided_slice/stack:output:0$fours/strided_slice/stack_1:output:0$fours/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
fours/strided_slicep
fours/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
fours/Reshape/shape/1?
fours/Reshape/shapePackfours/strided_slice:output:0fours/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
fours/Reshape/shape?
fours/ReshapeReshapefours/Cast:y:0fours/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
fours/Reshapeq
on_play/CastCast
features_4*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
on_play/Cast^
on_play/ShapeShapeon_play/Cast:y:0*
T0*
_output_shapes
:2
on_play/Shape?
on_play/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
on_play/strided_slice/stack?
on_play/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
on_play/strided_slice/stack_1?
on_play/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
on_play/strided_slice/stack_2?
on_play/strided_sliceStridedSliceon_play/Shape:output:0$on_play/strided_slice/stack:output:0&on_play/strided_slice/stack_1:output:0&on_play/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
on_play/strided_slicet
on_play/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
on_play/Reshape/shape/1?
on_play/Reshape/shapePackon_play/strided_slice:output:0 on_play/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
on_play/Reshape/shape?
on_play/ReshapeReshapeon_play/Cast:y:0on_play/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
on_play/Reshapek
	ones/CastCast
features_5*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
	ones/CastU

ones/ShapeShapeones/Cast:y:0*
T0*
_output_shapes
:2

ones/Shape~
ones/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
ones/strided_slice/stack?
ones/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
ones/strided_slice/stack_1?
ones/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
ones/strided_slice/stack_2?
ones/strided_sliceStridedSliceones/Shape:output:0!ones/strided_slice/stack:output:0#ones/strided_slice/stack_1:output:0#ones/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ones/strided_slicen
ones/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
ones/Reshape/shape/1?
ones/Reshape/shapePackones/strided_slice:output:0ones/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
ones/Reshape/shape?
ones/ReshapeReshapeones/Cast:y:0ones/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
ones/Reshape{
three_lander/CastCast
features_6*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
three_lander/Castm
three_lander/ShapeShapethree_lander/Cast:y:0*
T0*
_output_shapes
:2
three_lander/Shape?
 three_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 three_lander/strided_slice/stack?
"three_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"three_lander/strided_slice/stack_1?
"three_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"three_lander/strided_slice/stack_2?
three_lander/strided_sliceStridedSlicethree_lander/Shape:output:0)three_lander/strided_slice/stack:output:0+three_lander/strided_slice/stack_1:output:0+three_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
three_lander/strided_slice~
three_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
three_lander/Reshape/shape/1?
three_lander/Reshape/shapePack#three_lander/strided_slice:output:0%three_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
three_lander/Reshape/shape?
three_lander/ReshapeReshapethree_lander/Cast:y:0#three_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
three_lander/Reshapeo
threes/CastCast
features_7*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
threes/Cast[
threes/ShapeShapethrees/Cast:y:0*
T0*
_output_shapes
:2
threes/Shape?
threes/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
threes/strided_slice/stack?
threes/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
threes/strided_slice/stack_1?
threes/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
threes/strided_slice/stack_2?
threes/strided_sliceStridedSlicethrees/Shape:output:0#threes/strided_slice/stack:output:0%threes/strided_slice/stack_1:output:0%threes/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
threes/strided_slicer
threes/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
threes/Reshape/shape/1?
threes/Reshape/shapePackthrees/strided_slice:output:0threes/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
threes/Reshape/shape?
threes/ReshapeReshapethrees/Cast:y:0threes/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
threes/Reshapew
two_lander/CastCast
features_8*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
two_lander/Castg
two_lander/ShapeShapetwo_lander/Cast:y:0*
T0*
_output_shapes
:2
two_lander/Shape?
two_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
two_lander/strided_slice/stack?
 two_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 two_lander/strided_slice/stack_1?
 two_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 two_lander/strided_slice/stack_2?
two_lander/strided_sliceStridedSlicetwo_lander/Shape:output:0'two_lander/strided_slice/stack:output:0)two_lander/strided_slice/stack_1:output:0)two_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
two_lander/strided_slicez
two_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
two_lander/Reshape/shape/1?
two_lander/Reshape/shapePack!two_lander/strided_slice:output:0#two_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
two_lander/Reshape/shape?
two_lander/ReshapeReshapetwo_lander/Cast:y:0!two_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
two_lander/Reshapek
	twos/CastCast
features_9*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
	twos/CastU

twos/ShapeShapetwos/Cast:y:0*
T0*
_output_shapes
:2

twos/Shape~
twos/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
twos/strided_slice/stack?
twos/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
twos/strided_slice/stack_1?
twos/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
twos/strided_slice/stack_2?
twos/strided_sliceStridedSlicetwos/Shape:output:0!twos/strided_slice/stack:output:0#twos/strided_slice/stack_1:output:0#twos/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
twos/strided_slicen
twos/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
twos/Reshape/shape/1?
twos/Reshape/shapePacktwos/strided_slice:output:0twos/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
twos/Reshape/shape?
twos/ReshapeReshapetwos/Cast:y:0twos/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
twos/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2cards/Reshape:output:0five_lander/Reshape:output:0four_lander/Reshape:output:0fours/Reshape:output:0on_play/Reshape:output:0ones/Reshape:output:0three_lander/Reshape:output:0threes/Reshape:output:0two_lander/Reshape:output:0twos/Reshape:output:0concat/axis:output:0*
N
*
T0*'
_output_shapes
:?????????
2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
features
?
?
-__inference_sequential_layer_call_fn_11236291	
cards	
five_lander	
four_lander		
fours	
on_play	
ones	
three_lander	

threes	

two_lander	
twos	
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcardsfive_landerfour_landerfourson_playonesthree_landerthrees
two_landertwosunknown	unknown_0*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_112362662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namecards:TP
'
_output_shapes
:?????????
%
_user_specified_namefive_lander:TP
'
_output_shapes
:?????????
%
_user_specified_namefour_lander:NJ
'
_output_shapes
:?????????

_user_specified_namefours:PL
'
_output_shapes
:?????????
!
_user_specified_name	on_play:MI
'
_output_shapes
:?????????

_user_specified_nameones:UQ
'
_output_shapes
:?????????
&
_user_specified_namethree_lander:OK
'
_output_shapes
:?????????
 
_user_specified_namethrees:SO
'
_output_shapes
:?????????
$
_user_specified_name
two_lander:M	I
'
_output_shapes
:?????????

_user_specified_nametwos
?z
?
L__inference_dense_features_layer_call_and_return_conditional_losses_11236720
features_cards	
features_five_lander	
features_four_lander	
features_fours	
features_on_play	
features_ones	
features_three_lander	
features_threes	
features_two_lander	
features_twos	
identityq

cards/CastCastfeatures_cards*

DstT0*

SrcT0	*'
_output_shapes
:?????????2

cards/CastX
cards/ShapeShapecards/Cast:y:0*
T0*
_output_shapes
:2
cards/Shape?
cards/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cards/strided_slice/stack?
cards/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cards/strided_slice/stack_1?
cards/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cards/strided_slice/stack_2?
cards/strided_sliceStridedSlicecards/Shape:output:0"cards/strided_slice/stack:output:0$cards/strided_slice/stack_1:output:0$cards/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cards/strided_slicep
cards/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
cards/Reshape/shape/1?
cards/Reshape/shapePackcards/strided_slice:output:0cards/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
cards/Reshape/shape?
cards/ReshapeReshapecards/Cast:y:0cards/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
cards/Reshape?
five_lander/CastCastfeatures_five_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
five_lander/Castj
five_lander/ShapeShapefive_lander/Cast:y:0*
T0*
_output_shapes
:2
five_lander/Shape?
five_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
five_lander/strided_slice/stack?
!five_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!five_lander/strided_slice/stack_1?
!five_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!five_lander/strided_slice/stack_2?
five_lander/strided_sliceStridedSlicefive_lander/Shape:output:0(five_lander/strided_slice/stack:output:0*five_lander/strided_slice/stack_1:output:0*five_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
five_lander/strided_slice|
five_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
five_lander/Reshape/shape/1?
five_lander/Reshape/shapePack"five_lander/strided_slice:output:0$five_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
five_lander/Reshape/shape?
five_lander/ReshapeReshapefive_lander/Cast:y:0"five_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
five_lander/Reshape?
four_lander/CastCastfeatures_four_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
four_lander/Castj
four_lander/ShapeShapefour_lander/Cast:y:0*
T0*
_output_shapes
:2
four_lander/Shape?
four_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
four_lander/strided_slice/stack?
!four_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!four_lander/strided_slice/stack_1?
!four_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!four_lander/strided_slice/stack_2?
four_lander/strided_sliceStridedSlicefour_lander/Shape:output:0(four_lander/strided_slice/stack:output:0*four_lander/strided_slice/stack_1:output:0*four_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
four_lander/strided_slice|
four_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
four_lander/Reshape/shape/1?
four_lander/Reshape/shapePack"four_lander/strided_slice:output:0$four_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
four_lander/Reshape/shape?
four_lander/ReshapeReshapefour_lander/Cast:y:0"four_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
four_lander/Reshapeq

fours/CastCastfeatures_fours*

DstT0*

SrcT0	*'
_output_shapes
:?????????2

fours/CastX
fours/ShapeShapefours/Cast:y:0*
T0*
_output_shapes
:2
fours/Shape?
fours/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
fours/strided_slice/stack?
fours/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
fours/strided_slice/stack_1?
fours/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
fours/strided_slice/stack_2?
fours/strided_sliceStridedSlicefours/Shape:output:0"fours/strided_slice/stack:output:0$fours/strided_slice/stack_1:output:0$fours/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
fours/strided_slicep
fours/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
fours/Reshape/shape/1?
fours/Reshape/shapePackfours/strided_slice:output:0fours/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
fours/Reshape/shape?
fours/ReshapeReshapefours/Cast:y:0fours/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
fours/Reshapew
on_play/CastCastfeatures_on_play*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
on_play/Cast^
on_play/ShapeShapeon_play/Cast:y:0*
T0*
_output_shapes
:2
on_play/Shape?
on_play/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
on_play/strided_slice/stack?
on_play/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
on_play/strided_slice/stack_1?
on_play/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
on_play/strided_slice/stack_2?
on_play/strided_sliceStridedSliceon_play/Shape:output:0$on_play/strided_slice/stack:output:0&on_play/strided_slice/stack_1:output:0&on_play/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
on_play/strided_slicet
on_play/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
on_play/Reshape/shape/1?
on_play/Reshape/shapePackon_play/strided_slice:output:0 on_play/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
on_play/Reshape/shape?
on_play/ReshapeReshapeon_play/Cast:y:0on_play/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
on_play/Reshapen
	ones/CastCastfeatures_ones*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
	ones/CastU

ones/ShapeShapeones/Cast:y:0*
T0*
_output_shapes
:2

ones/Shape~
ones/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
ones/strided_slice/stack?
ones/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
ones/strided_slice/stack_1?
ones/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
ones/strided_slice/stack_2?
ones/strided_sliceStridedSliceones/Shape:output:0!ones/strided_slice/stack:output:0#ones/strided_slice/stack_1:output:0#ones/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ones/strided_slicen
ones/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
ones/Reshape/shape/1?
ones/Reshape/shapePackones/strided_slice:output:0ones/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
ones/Reshape/shape?
ones/ReshapeReshapeones/Cast:y:0ones/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
ones/Reshape?
three_lander/CastCastfeatures_three_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
three_lander/Castm
three_lander/ShapeShapethree_lander/Cast:y:0*
T0*
_output_shapes
:2
three_lander/Shape?
 three_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 three_lander/strided_slice/stack?
"three_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"three_lander/strided_slice/stack_1?
"three_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"three_lander/strided_slice/stack_2?
three_lander/strided_sliceStridedSlicethree_lander/Shape:output:0)three_lander/strided_slice/stack:output:0+three_lander/strided_slice/stack_1:output:0+three_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
three_lander/strided_slice~
three_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
three_lander/Reshape/shape/1?
three_lander/Reshape/shapePack#three_lander/strided_slice:output:0%three_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
three_lander/Reshape/shape?
three_lander/ReshapeReshapethree_lander/Cast:y:0#three_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
three_lander/Reshapet
threes/CastCastfeatures_threes*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
threes/Cast[
threes/ShapeShapethrees/Cast:y:0*
T0*
_output_shapes
:2
threes/Shape?
threes/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
threes/strided_slice/stack?
threes/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
threes/strided_slice/stack_1?
threes/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
threes/strided_slice/stack_2?
threes/strided_sliceStridedSlicethrees/Shape:output:0#threes/strided_slice/stack:output:0%threes/strided_slice/stack_1:output:0%threes/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
threes/strided_slicer
threes/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
threes/Reshape/shape/1?
threes/Reshape/shapePackthrees/strided_slice:output:0threes/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
threes/Reshape/shape?
threes/ReshapeReshapethrees/Cast:y:0threes/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
threes/Reshape?
two_lander/CastCastfeatures_two_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
two_lander/Castg
two_lander/ShapeShapetwo_lander/Cast:y:0*
T0*
_output_shapes
:2
two_lander/Shape?
two_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
two_lander/strided_slice/stack?
 two_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 two_lander/strided_slice/stack_1?
 two_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 two_lander/strided_slice/stack_2?
two_lander/strided_sliceStridedSlicetwo_lander/Shape:output:0'two_lander/strided_slice/stack:output:0)two_lander/strided_slice/stack_1:output:0)two_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
two_lander/strided_slicez
two_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
two_lander/Reshape/shape/1?
two_lander/Reshape/shapePack!two_lander/strided_slice:output:0#two_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
two_lander/Reshape/shape?
two_lander/ReshapeReshapetwo_lander/Cast:y:0!two_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
two_lander/Reshapen
	twos/CastCastfeatures_twos*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
	twos/CastU

twos/ShapeShapetwos/Cast:y:0*
T0*
_output_shapes
:2

twos/Shape~
twos/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
twos/strided_slice/stack?
twos/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
twos/strided_slice/stack_1?
twos/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
twos/strided_slice/stack_2?
twos/strided_sliceStridedSlicetwos/Shape:output:0!twos/strided_slice/stack:output:0#twos/strided_slice/stack_1:output:0#twos/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
twos/strided_slicen
twos/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
twos/Reshape/shape/1?
twos/Reshape/shapePacktwos/strided_slice:output:0twos/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
twos/Reshape/shape?
twos/ReshapeReshapetwos/Cast:y:0twos/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
twos/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2cards/Reshape:output:0five_lander/Reshape:output:0four_lander/Reshape:output:0fours/Reshape:output:0on_play/Reshape:output:0ones/Reshape:output:0three_lander/Reshape:output:0threes/Reshape:output:0two_lander/Reshape:output:0twos/Reshape:output:0concat/axis:output:0*
N
*
T0*'
_output_shapes
:?????????
2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:W S
'
_output_shapes
:?????????
(
_user_specified_namefeatures/cards:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/five_lander:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/four_lander:WS
'
_output_shapes
:?????????
(
_user_specified_namefeatures/fours:YU
'
_output_shapes
:?????????
*
_user_specified_namefeatures/on_play:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/ones:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/three_lander:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/threes:\X
'
_output_shapes
:?????????
-
_user_specified_namefeatures/two_lander:V	R
'
_output_shapes
:?????????
'
_user_specified_namefeatures/twos
??
?
H__inference_sequential_layer_call_and_return_conditional_losses_11236467
inputs_cards	
inputs_five_lander	
inputs_four_lander	
inputs_fours	
inputs_on_play	
inputs_ones	
inputs_three_lander	
inputs_threes	
inputs_two_lander	
inputs_twos	6
$dense_matmul_readvariableop_resource:
3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense_features/cards/CastCastinputs_cards*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
dense_features/cards/Cast?
dense_features/cards/ShapeShapedense_features/cards/Cast:y:0*
T0*
_output_shapes
:2
dense_features/cards/Shape?
(dense_features/cards/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(dense_features/cards/strided_slice/stack?
*dense_features/cards/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features/cards/strided_slice/stack_1?
*dense_features/cards/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features/cards/strided_slice/stack_2?
"dense_features/cards/strided_sliceStridedSlice#dense_features/cards/Shape:output:01dense_features/cards/strided_slice/stack:output:03dense_features/cards/strided_slice/stack_1:output:03dense_features/cards/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"dense_features/cards/strided_slice?
$dense_features/cards/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$dense_features/cards/Reshape/shape/1?
"dense_features/cards/Reshape/shapePack+dense_features/cards/strided_slice:output:0-dense_features/cards/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2$
"dense_features/cards/Reshape/shape?
dense_features/cards/ReshapeReshapedense_features/cards/Cast:y:0+dense_features/cards/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dense_features/cards/Reshape?
dense_features/five_lander/CastCastinputs_five_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2!
dense_features/five_lander/Cast?
 dense_features/five_lander/ShapeShape#dense_features/five_lander/Cast:y:0*
T0*
_output_shapes
:2"
 dense_features/five_lander/Shape?
.dense_features/five_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.dense_features/five_lander/strided_slice/stack?
0dense_features/five_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0dense_features/five_lander/strided_slice/stack_1?
0dense_features/five_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0dense_features/five_lander/strided_slice/stack_2?
(dense_features/five_lander/strided_sliceStridedSlice)dense_features/five_lander/Shape:output:07dense_features/five_lander/strided_slice/stack:output:09dense_features/five_lander/strided_slice/stack_1:output:09dense_features/five_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(dense_features/five_lander/strided_slice?
*dense_features/five_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*dense_features/five_lander/Reshape/shape/1?
(dense_features/five_lander/Reshape/shapePack1dense_features/five_lander/strided_slice:output:03dense_features/five_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2*
(dense_features/five_lander/Reshape/shape?
"dense_features/five_lander/ReshapeReshape#dense_features/five_lander/Cast:y:01dense_features/five_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2$
"dense_features/five_lander/Reshape?
dense_features/four_lander/CastCastinputs_four_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2!
dense_features/four_lander/Cast?
 dense_features/four_lander/ShapeShape#dense_features/four_lander/Cast:y:0*
T0*
_output_shapes
:2"
 dense_features/four_lander/Shape?
.dense_features/four_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.dense_features/four_lander/strided_slice/stack?
0dense_features/four_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0dense_features/four_lander/strided_slice/stack_1?
0dense_features/four_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0dense_features/four_lander/strided_slice/stack_2?
(dense_features/four_lander/strided_sliceStridedSlice)dense_features/four_lander/Shape:output:07dense_features/four_lander/strided_slice/stack:output:09dense_features/four_lander/strided_slice/stack_1:output:09dense_features/four_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(dense_features/four_lander/strided_slice?
*dense_features/four_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*dense_features/four_lander/Reshape/shape/1?
(dense_features/four_lander/Reshape/shapePack1dense_features/four_lander/strided_slice:output:03dense_features/four_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2*
(dense_features/four_lander/Reshape/shape?
"dense_features/four_lander/ReshapeReshape#dense_features/four_lander/Cast:y:01dense_features/four_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2$
"dense_features/four_lander/Reshape?
dense_features/fours/CastCastinputs_fours*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
dense_features/fours/Cast?
dense_features/fours/ShapeShapedense_features/fours/Cast:y:0*
T0*
_output_shapes
:2
dense_features/fours/Shape?
(dense_features/fours/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(dense_features/fours/strided_slice/stack?
*dense_features/fours/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features/fours/strided_slice/stack_1?
*dense_features/fours/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features/fours/strided_slice/stack_2?
"dense_features/fours/strided_sliceStridedSlice#dense_features/fours/Shape:output:01dense_features/fours/strided_slice/stack:output:03dense_features/fours/strided_slice/stack_1:output:03dense_features/fours/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"dense_features/fours/strided_slice?
$dense_features/fours/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$dense_features/fours/Reshape/shape/1?
"dense_features/fours/Reshape/shapePack+dense_features/fours/strided_slice:output:0-dense_features/fours/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2$
"dense_features/fours/Reshape/shape?
dense_features/fours/ReshapeReshapedense_features/fours/Cast:y:0+dense_features/fours/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dense_features/fours/Reshape?
dense_features/on_play/CastCastinputs_on_play*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
dense_features/on_play/Cast?
dense_features/on_play/ShapeShapedense_features/on_play/Cast:y:0*
T0*
_output_shapes
:2
dense_features/on_play/Shape?
*dense_features/on_play/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*dense_features/on_play/strided_slice/stack?
,dense_features/on_play/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features/on_play/strided_slice/stack_1?
,dense_features/on_play/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features/on_play/strided_slice/stack_2?
$dense_features/on_play/strided_sliceStridedSlice%dense_features/on_play/Shape:output:03dense_features/on_play/strided_slice/stack:output:05dense_features/on_play/strided_slice/stack_1:output:05dense_features/on_play/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$dense_features/on_play/strided_slice?
&dense_features/on_play/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&dense_features/on_play/Reshape/shape/1?
$dense_features/on_play/Reshape/shapePack-dense_features/on_play/strided_slice:output:0/dense_features/on_play/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$dense_features/on_play/Reshape/shape?
dense_features/on_play/ReshapeReshapedense_features/on_play/Cast:y:0-dense_features/on_play/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2 
dense_features/on_play/Reshape?
dense_features/ones/CastCastinputs_ones*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
dense_features/ones/Cast?
dense_features/ones/ShapeShapedense_features/ones/Cast:y:0*
T0*
_output_shapes
:2
dense_features/ones/Shape?
'dense_features/ones/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'dense_features/ones/strided_slice/stack?
)dense_features/ones/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)dense_features/ones/strided_slice/stack_1?
)dense_features/ones/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)dense_features/ones/strided_slice/stack_2?
!dense_features/ones/strided_sliceStridedSlice"dense_features/ones/Shape:output:00dense_features/ones/strided_slice/stack:output:02dense_features/ones/strided_slice/stack_1:output:02dense_features/ones/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!dense_features/ones/strided_slice?
#dense_features/ones/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#dense_features/ones/Reshape/shape/1?
!dense_features/ones/Reshape/shapePack*dense_features/ones/strided_slice:output:0,dense_features/ones/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2#
!dense_features/ones/Reshape/shape?
dense_features/ones/ReshapeReshapedense_features/ones/Cast:y:0*dense_features/ones/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dense_features/ones/Reshape?
 dense_features/three_lander/CastCastinputs_three_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2"
 dense_features/three_lander/Cast?
!dense_features/three_lander/ShapeShape$dense_features/three_lander/Cast:y:0*
T0*
_output_shapes
:2#
!dense_features/three_lander/Shape?
/dense_features/three_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_features/three_lander/strided_slice/stack?
1dense_features/three_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_features/three_lander/strided_slice/stack_1?
1dense_features/three_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_features/three_lander/strided_slice/stack_2?
)dense_features/three_lander/strided_sliceStridedSlice*dense_features/three_lander/Shape:output:08dense_features/three_lander/strided_slice/stack:output:0:dense_features/three_lander/strided_slice/stack_1:output:0:dense_features/three_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_features/three_lander/strided_slice?
+dense_features/three_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+dense_features/three_lander/Reshape/shape/1?
)dense_features/three_lander/Reshape/shapePack2dense_features/three_lander/strided_slice:output:04dense_features/three_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2+
)dense_features/three_lander/Reshape/shape?
#dense_features/three_lander/ReshapeReshape$dense_features/three_lander/Cast:y:02dense_features/three_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2%
#dense_features/three_lander/Reshape?
dense_features/threes/CastCastinputs_threes*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
dense_features/threes/Cast?
dense_features/threes/ShapeShapedense_features/threes/Cast:y:0*
T0*
_output_shapes
:2
dense_features/threes/Shape?
)dense_features/threes/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)dense_features/threes/strided_slice/stack?
+dense_features/threes/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features/threes/strided_slice/stack_1?
+dense_features/threes/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features/threes/strided_slice/stack_2?
#dense_features/threes/strided_sliceStridedSlice$dense_features/threes/Shape:output:02dense_features/threes/strided_slice/stack:output:04dense_features/threes/strided_slice/stack_1:output:04dense_features/threes/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#dense_features/threes/strided_slice?
%dense_features/threes/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%dense_features/threes/Reshape/shape/1?
#dense_features/threes/Reshape/shapePack,dense_features/threes/strided_slice:output:0.dense_features/threes/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#dense_features/threes/Reshape/shape?
dense_features/threes/ReshapeReshapedense_features/threes/Cast:y:0,dense_features/threes/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dense_features/threes/Reshape?
dense_features/two_lander/CastCastinputs_two_lander*

DstT0*

SrcT0	*'
_output_shapes
:?????????2 
dense_features/two_lander/Cast?
dense_features/two_lander/ShapeShape"dense_features/two_lander/Cast:y:0*
T0*
_output_shapes
:2!
dense_features/two_lander/Shape?
-dense_features/two_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense_features/two_lander/strided_slice/stack?
/dense_features/two_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense_features/two_lander/strided_slice/stack_1?
/dense_features/two_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense_features/two_lander/strided_slice/stack_2?
'dense_features/two_lander/strided_sliceStridedSlice(dense_features/two_lander/Shape:output:06dense_features/two_lander/strided_slice/stack:output:08dense_features/two_lander/strided_slice/stack_1:output:08dense_features/two_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense_features/two_lander/strided_slice?
)dense_features/two_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)dense_features/two_lander/Reshape/shape/1?
'dense_features/two_lander/Reshape/shapePack0dense_features/two_lander/strided_slice:output:02dense_features/two_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2)
'dense_features/two_lander/Reshape/shape?
!dense_features/two_lander/ReshapeReshape"dense_features/two_lander/Cast:y:00dense_features/two_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2#
!dense_features/two_lander/Reshape?
dense_features/twos/CastCastinputs_twos*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
dense_features/twos/Cast?
dense_features/twos/ShapeShapedense_features/twos/Cast:y:0*
T0*
_output_shapes
:2
dense_features/twos/Shape?
'dense_features/twos/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'dense_features/twos/strided_slice/stack?
)dense_features/twos/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)dense_features/twos/strided_slice/stack_1?
)dense_features/twos/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)dense_features/twos/strided_slice/stack_2?
!dense_features/twos/strided_sliceStridedSlice"dense_features/twos/Shape:output:00dense_features/twos/strided_slice/stack:output:02dense_features/twos/strided_slice/stack_1:output:02dense_features/twos/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!dense_features/twos/strided_slice?
#dense_features/twos/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#dense_features/twos/Reshape/shape/1?
!dense_features/twos/Reshape/shapePack*dense_features/twos/strided_slice:output:0,dense_features/twos/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2#
!dense_features/twos/Reshape/shape?
dense_features/twos/ReshapeReshapedense_features/twos/Cast:y:0*dense_features/twos/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dense_features/twos/Reshape?
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
dense_features/concat/axis?
dense_features/concatConcatV2%dense_features/cards/Reshape:output:0+dense_features/five_lander/Reshape:output:0+dense_features/four_lander/Reshape:output:0%dense_features/fours/Reshape:output:0'dense_features/on_play/Reshape:output:0$dense_features/ones/Reshape:output:0,dense_features/three_lander/Reshape:output:0&dense_features/threes/Reshape:output:0*dense_features/two_lander/Reshape:output:0$dense_features/twos/Reshape:output:0#dense_features/concat/axis:output:0*
N
*
T0*'
_output_shapes
:?????????
2
dense_features/concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMuldense_features/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense/Sigmoidl
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:U Q
'
_output_shapes
:?????????
&
_user_specified_nameinputs/cards:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/five_lander:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/four_lander:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/fours:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/on_play:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/ones:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/three_lander:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/threes:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/two_lander:T	P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/twos
?
?
C__inference_dense_layer_call_and_return_conditional_losses_11236060

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
H__inference_sequential_layer_call_and_return_conditional_losses_11236067

inputs	
inputs_1	
inputs_2	
inputs_3	
inputs_4	
inputs_5	
inputs_6	
inputs_7	
inputs_8	
inputs_9	 
dense_11236061:

dense_11236063:
identity??dense/StatefulPartitionedCall?
dense_features/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9*
Tin
2
										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_dense_features_layer_call_and_return_conditional_losses_112360472 
dense_features/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0dense_11236061dense_11236063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_112360602
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityn
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_dense_layer_call_fn_11236873

inputs
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_112360602
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_11236355	
cards	
five_lander	
four_lander		
fours	
on_play	
ones	
three_lander	

threes	

two_lander	
twos	
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcardsfive_landerfour_landerfourson_playonesthree_landerthrees
two_landertwosunknown	unknown_0*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_112359172
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_namecards:TP
'
_output_shapes
:?????????
%
_user_specified_namefive_lander:TP
'
_output_shapes
:?????????
%
_user_specified_namefour_lander:NJ
'
_output_shapes
:?????????

_user_specified_namefours:PL
'
_output_shapes
:?????????
!
_user_specified_name	on_play:MI
'
_output_shapes
:?????????

_user_specified_nameones:UQ
'
_output_shapes
:?????????
&
_user_specified_namethree_lander:OK
'
_output_shapes
:?????????
 
_user_specified_namethrees:SO
'
_output_shapes
:?????????
$
_user_specified_name
two_lander:M	I
'
_output_shapes
:?????????

_user_specified_nametwos
?x
?
L__inference_dense_features_layer_call_and_return_conditional_losses_11236047
features	

features_1	

features_2	

features_3	

features_4	

features_5	

features_6	

features_7	

features_8	

features_9	
identityk

cards/CastCastfeatures*

DstT0*

SrcT0	*'
_output_shapes
:?????????2

cards/CastX
cards/ShapeShapecards/Cast:y:0*
T0*
_output_shapes
:2
cards/Shape?
cards/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cards/strided_slice/stack?
cards/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cards/strided_slice/stack_1?
cards/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cards/strided_slice/stack_2?
cards/strided_sliceStridedSlicecards/Shape:output:0"cards/strided_slice/stack:output:0$cards/strided_slice/stack_1:output:0$cards/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cards/strided_slicep
cards/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
cards/Reshape/shape/1?
cards/Reshape/shapePackcards/strided_slice:output:0cards/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
cards/Reshape/shape?
cards/ReshapeReshapecards/Cast:y:0cards/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
cards/Reshapey
five_lander/CastCast
features_1*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
five_lander/Castj
five_lander/ShapeShapefive_lander/Cast:y:0*
T0*
_output_shapes
:2
five_lander/Shape?
five_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
five_lander/strided_slice/stack?
!five_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!five_lander/strided_slice/stack_1?
!five_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!five_lander/strided_slice/stack_2?
five_lander/strided_sliceStridedSlicefive_lander/Shape:output:0(five_lander/strided_slice/stack:output:0*five_lander/strided_slice/stack_1:output:0*five_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
five_lander/strided_slice|
five_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
five_lander/Reshape/shape/1?
five_lander/Reshape/shapePack"five_lander/strided_slice:output:0$five_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
five_lander/Reshape/shape?
five_lander/ReshapeReshapefive_lander/Cast:y:0"five_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
five_lander/Reshapey
four_lander/CastCast
features_2*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
four_lander/Castj
four_lander/ShapeShapefour_lander/Cast:y:0*
T0*
_output_shapes
:2
four_lander/Shape?
four_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
four_lander/strided_slice/stack?
!four_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!four_lander/strided_slice/stack_1?
!four_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!four_lander/strided_slice/stack_2?
four_lander/strided_sliceStridedSlicefour_lander/Shape:output:0(four_lander/strided_slice/stack:output:0*four_lander/strided_slice/stack_1:output:0*four_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
four_lander/strided_slice|
four_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
four_lander/Reshape/shape/1?
four_lander/Reshape/shapePack"four_lander/strided_slice:output:0$four_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
four_lander/Reshape/shape?
four_lander/ReshapeReshapefour_lander/Cast:y:0"four_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
four_lander/Reshapem

fours/CastCast
features_3*

DstT0*

SrcT0	*'
_output_shapes
:?????????2

fours/CastX
fours/ShapeShapefours/Cast:y:0*
T0*
_output_shapes
:2
fours/Shape?
fours/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
fours/strided_slice/stack?
fours/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
fours/strided_slice/stack_1?
fours/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
fours/strided_slice/stack_2?
fours/strided_sliceStridedSlicefours/Shape:output:0"fours/strided_slice/stack:output:0$fours/strided_slice/stack_1:output:0$fours/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
fours/strided_slicep
fours/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
fours/Reshape/shape/1?
fours/Reshape/shapePackfours/strided_slice:output:0fours/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
fours/Reshape/shape?
fours/ReshapeReshapefours/Cast:y:0fours/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
fours/Reshapeq
on_play/CastCast
features_4*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
on_play/Cast^
on_play/ShapeShapeon_play/Cast:y:0*
T0*
_output_shapes
:2
on_play/Shape?
on_play/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
on_play/strided_slice/stack?
on_play/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
on_play/strided_slice/stack_1?
on_play/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
on_play/strided_slice/stack_2?
on_play/strided_sliceStridedSliceon_play/Shape:output:0$on_play/strided_slice/stack:output:0&on_play/strided_slice/stack_1:output:0&on_play/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
on_play/strided_slicet
on_play/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
on_play/Reshape/shape/1?
on_play/Reshape/shapePackon_play/strided_slice:output:0 on_play/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
on_play/Reshape/shape?
on_play/ReshapeReshapeon_play/Cast:y:0on_play/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
on_play/Reshapek
	ones/CastCast
features_5*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
	ones/CastU

ones/ShapeShapeones/Cast:y:0*
T0*
_output_shapes
:2

ones/Shape~
ones/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
ones/strided_slice/stack?
ones/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
ones/strided_slice/stack_1?
ones/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
ones/strided_slice/stack_2?
ones/strided_sliceStridedSliceones/Shape:output:0!ones/strided_slice/stack:output:0#ones/strided_slice/stack_1:output:0#ones/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ones/strided_slicen
ones/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
ones/Reshape/shape/1?
ones/Reshape/shapePackones/strided_slice:output:0ones/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
ones/Reshape/shape?
ones/ReshapeReshapeones/Cast:y:0ones/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
ones/Reshape{
three_lander/CastCast
features_6*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
three_lander/Castm
three_lander/ShapeShapethree_lander/Cast:y:0*
T0*
_output_shapes
:2
three_lander/Shape?
 three_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 three_lander/strided_slice/stack?
"three_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"three_lander/strided_slice/stack_1?
"three_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"three_lander/strided_slice/stack_2?
three_lander/strided_sliceStridedSlicethree_lander/Shape:output:0)three_lander/strided_slice/stack:output:0+three_lander/strided_slice/stack_1:output:0+three_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
three_lander/strided_slice~
three_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
three_lander/Reshape/shape/1?
three_lander/Reshape/shapePack#three_lander/strided_slice:output:0%three_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
three_lander/Reshape/shape?
three_lander/ReshapeReshapethree_lander/Cast:y:0#three_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
three_lander/Reshapeo
threes/CastCast
features_7*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
threes/Cast[
threes/ShapeShapethrees/Cast:y:0*
T0*
_output_shapes
:2
threes/Shape?
threes/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
threes/strided_slice/stack?
threes/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
threes/strided_slice/stack_1?
threes/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
threes/strided_slice/stack_2?
threes/strided_sliceStridedSlicethrees/Shape:output:0#threes/strided_slice/stack:output:0%threes/strided_slice/stack_1:output:0%threes/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
threes/strided_slicer
threes/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
threes/Reshape/shape/1?
threes/Reshape/shapePackthrees/strided_slice:output:0threes/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
threes/Reshape/shape?
threes/ReshapeReshapethrees/Cast:y:0threes/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
threes/Reshapew
two_lander/CastCast
features_8*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
two_lander/Castg
two_lander/ShapeShapetwo_lander/Cast:y:0*
T0*
_output_shapes
:2
two_lander/Shape?
two_lander/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
two_lander/strided_slice/stack?
 two_lander/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 two_lander/strided_slice/stack_1?
 two_lander/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 two_lander/strided_slice/stack_2?
two_lander/strided_sliceStridedSlicetwo_lander/Shape:output:0'two_lander/strided_slice/stack:output:0)two_lander/strided_slice/stack_1:output:0)two_lander/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
two_lander/strided_slicez
two_lander/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
two_lander/Reshape/shape/1?
two_lander/Reshape/shapePack!two_lander/strided_slice:output:0#two_lander/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
two_lander/Reshape/shape?
two_lander/ReshapeReshapetwo_lander/Cast:y:0!two_lander/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
two_lander/Reshapek
	twos/CastCast
features_9*

DstT0*

SrcT0	*'
_output_shapes
:?????????2
	twos/CastU

twos/ShapeShapetwos/Cast:y:0*
T0*
_output_shapes
:2

twos/Shape~
twos/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
twos/strided_slice/stack?
twos/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
twos/strided_slice/stack_1?
twos/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
twos/strided_slice/stack_2?
twos/strided_sliceStridedSlicetwos/Shape:output:0!twos/strided_slice/stack:output:0#twos/strided_slice/stack_1:output:0#twos/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
twos/strided_slicen
twos/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
twos/Reshape/shape/1?
twos/Reshape/shapePacktwos/strided_slice:output:0twos/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
twos/Reshape/shape?
twos/ReshapeReshapetwos/Cast:y:0twos/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
twos/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2cards/Reshape:output:0five_lander/Reshape:output:0four_lander/Reshape:output:0fours/Reshape:output:0on_play/Reshape:output:0ones/Reshape:output:0three_lander/Reshape:output:0threes/Reshape:output:0two_lander/Reshape:output:0twos/Reshape:output:0concat/axis:output:0*
N
*
T0*'
_output_shapes
:?????????
2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
features
?
?
-__inference_sequential_layer_call_fn_11236615
inputs_cards	
inputs_five_lander	
inputs_four_lander	
inputs_fours	
inputs_on_play	
inputs_ones	
inputs_three_lander	
inputs_threes	
inputs_two_lander	
inputs_twos	
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_cardsinputs_five_landerinputs_four_landerinputs_foursinputs_on_playinputs_onesinputs_three_landerinputs_threesinputs_two_landerinputs_twosunknown	unknown_0*
Tin
2										*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_112362662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_nameinputs/cards:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/five_lander:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/four_lander:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/fours:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/on_play:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/ones:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/three_lander:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/threes:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/two_lander:T	P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/twos
?
?
C__inference_dense_layer_call_and_return_conditional_losses_11236864

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?)
?
!__inference__traced_save_11236950
file_prefix6
2savev2_sequential_dense_kernel_read_readvariableop4
0savev2_sequential_dense_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableopB
>savev2_rmsprop_sequential_dense_kernel_rms_read_readvariableop@
<savev2_rmsprop_sequential_dense_bias_rms_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_sequential_dense_kernel_read_readvariableop0savev2_sequential_dense_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop>savev2_rmsprop_sequential_dense_kernel_rms_read_readvariableop<savev2_rmsprop_sequential_dense_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*M
_input_shapes<
:: :
:: : : : : : : : : : : :
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
cards.
serving_default_cards:0	?????????
C
five_lander4
serving_default_five_lander:0	?????????
C
four_lander4
serving_default_four_lander:0	?????????
7
fours.
serving_default_fours:0	?????????
;
on_play0
serving_default_on_play:0	?????????
5
ones-
serving_default_ones:0	?????????
E
three_lander5
serving_default_three_lander:0	?????????
9
threes/
serving_default_threes:0	?????????
A

two_lander3
serving_default_two_lander:0	?????????
5
twos-
serving_default_twos:0	?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?|
?
layer-0
layer_with_weights-0
layer-1
	optimizer
_build_input_shape
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
*=&call_and_return_all_conditional_losses
>_default_save_signature
?__call__"
_tf_keras_sequential
?

_feature_columns

_resources
trainable_variables
	variables
regularization_losses
	keras_api
*@&call_and_return_all_conditional_losses
A__call__"
_tf_keras_layer
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*B&call_and_return_all_conditional_losses
C__call__"
_tf_keras_layer
h
iter
	decay
learning_rate
momentum
rho	rms;	rms<"
	optimizer
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables

layers
regularization_losses
non_trainable_variables
layer_regularization_losses
metrics
layer_metrics
?__call__
>_default_save_signature
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
,
Dserving_default"
signature_map
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
 metrics

!layers
	variables
regularization_losses
"layer_regularization_losses
#non_trainable_variables
$layer_metrics
A__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
):'
2sequential/dense/kernel
#:!2sequential/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
%metrics

&layers
	variables
regularization_losses
'layer_regularization_losses
(non_trainable_variables
)layer_metrics
C__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	-total
	.count
/	variables
0	keras_api"
_tf_keras_metric
^
	1total
	2count
3
_fn_kwargs
4	variables
5	keras_api"
_tf_keras_metric
^
	6total
	7count
8
_fn_kwargs
9	variables
:	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
-0
.1"
trackable_list_wrapper
-
/	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
-
4	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
-
9	variables"
_generic_user_object
3:1
2#RMSprop/sequential/dense/kernel/rms
-:+2!RMSprop/sequential/dense/bias/rms
?2?
H__inference_sequential_layer_call_and_return_conditional_losses_11236467
H__inference_sequential_layer_call_and_return_conditional_losses_11236579
H__inference_sequential_layer_call_and_return_conditional_losses_11236310
H__inference_sequential_layer_call_and_return_conditional_losses_11236329?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference__wrapped_model_11235917cardsfive_landerfour_landerfourson_playonesthree_landerthrees
two_landertwos
"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_sequential_layer_call_fn_11236074
-__inference_sequential_layer_call_fn_11236597
-__inference_sequential_layer_call_fn_11236615
-__inference_sequential_layer_call_fn_11236291?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_dense_features_layer_call_and_return_conditional_losses_11236720
L__inference_dense_features_layer_call_and_return_conditional_losses_11236825?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_dense_features_layer_call_fn_11236839
1__inference_dense_features_layer_call_fn_11236853?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dense_layer_call_and_return_conditional_losses_11236864?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_layer_call_fn_11236873?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_11236355cardsfive_landerfour_landerfourson_playonesthree_landerthrees
two_landertwos"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_11235917????
???
???
(
cards?
cards?????????	
4
five_lander%?"
five_lander?????????	
4
four_lander%?"
four_lander?????????	
(
fours?
fours?????????	
,
on_play!?
on_play?????????	
&
ones?
ones?????????	
6
three_lander&?#
three_lander?????????	
*
threes ?
threes?????????	
2

two_lander$?!

two_lander?????????	
&
twos?
twos?????????	
? "3?0
.
output_1"?
output_1??????????
L__inference_dense_features_layer_call_and_return_conditional_losses_11236720????
???
???
1
cards(?%
features/cards?????????	
=
five_lander.?+
features/five_lander?????????	
=
four_lander.?+
features/four_lander?????????	
1
fours(?%
features/fours?????????	
5
on_play*?'
features/on_play?????????	
/
ones'?$
features/ones?????????	
?
three_lander/?,
features/three_lander?????????	
3
threes)?&
features/threes?????????	
;

two_lander-?*
features/two_lander?????????	
/
twos'?$
features/twos?????????	

 
p 
? "%?"
?
0?????????

? ?
L__inference_dense_features_layer_call_and_return_conditional_losses_11236825????
???
???
1
cards(?%
features/cards?????????	
=
five_lander.?+
features/five_lander?????????	
=
four_lander.?+
features/four_lander?????????	
1
fours(?%
features/fours?????????	
5
on_play*?'
features/on_play?????????	
/
ones'?$
features/ones?????????	
?
three_lander/?,
features/three_lander?????????	
3
threes)?&
features/threes?????????	
;

two_lander-?*
features/two_lander?????????	
/
twos'?$
features/twos?????????	

 
p
? "%?"
?
0?????????

? ?
1__inference_dense_features_layer_call_fn_11236839????
???
???
1
cards(?%
features/cards?????????	
=
five_lander.?+
features/five_lander?????????	
=
four_lander.?+
features/four_lander?????????	
1
fours(?%
features/fours?????????	
5
on_play*?'
features/on_play?????????	
/
ones'?$
features/ones?????????	
?
three_lander/?,
features/three_lander?????????	
3
threes)?&
features/threes?????????	
;

two_lander-?*
features/two_lander?????????	
/
twos'?$
features/twos?????????	

 
p 
? "??????????
?
1__inference_dense_features_layer_call_fn_11236853????
???
???
1
cards(?%
features/cards?????????	
=
five_lander.?+
features/five_lander?????????	
=
four_lander.?+
features/four_lander?????????	
1
fours(?%
features/fours?????????	
5
on_play*?'
features/on_play?????????	
/
ones'?$
features/ones?????????	
?
three_lander/?,
features/three_lander?????????	
3
threes)?&
features/threes?????????	
;

two_lander-?*
features/two_lander?????????	
/
twos'?$
features/twos?????????	

 
p
? "??????????
?
C__inference_dense_layer_call_and_return_conditional_losses_11236864\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? {
(__inference_dense_layer_call_fn_11236873O/?,
%?"
 ?
inputs?????????

? "???????????
H__inference_sequential_layer_call_and_return_conditional_losses_11236310????
???
???
(
cards?
cards?????????	
4
five_lander%?"
five_lander?????????	
4
four_lander%?"
four_lander?????????	
(
fours?
fours?????????	
,
on_play!?
on_play?????????	
&
ones?
ones?????????	
6
three_lander&?#
three_lander?????????	
*
threes ?
threes?????????	
2

two_lander$?!

two_lander?????????	
&
twos?
twos?????????	
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_11236329????
???
???
(
cards?
cards?????????	
4
five_lander%?"
five_lander?????????	
4
four_lander%?"
four_lander?????????	
(
fours?
fours?????????	
,
on_play!?
on_play?????????	
&
ones?
ones?????????	
6
three_lander&?#
three_lander?????????	
*
threes ?
threes?????????	
2

two_lander$?!

two_lander?????????	
&
twos?
twos?????????	
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_11236467????
???
???
/
cards&?#
inputs/cards?????????	
;
five_lander,?)
inputs/five_lander?????????	
;
four_lander,?)
inputs/four_lander?????????	
/
fours&?#
inputs/fours?????????	
3
on_play(?%
inputs/on_play?????????	
-
ones%?"
inputs/ones?????????	
=
three_lander-?*
inputs/three_lander?????????	
1
threes'?$
inputs/threes?????????	
9

two_lander+?(
inputs/two_lander?????????	
-
twos%?"
inputs/twos?????????	
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_layer_call_and_return_conditional_losses_11236579????
???
???
/
cards&?#
inputs/cards?????????	
;
five_lander,?)
inputs/five_lander?????????	
;
four_lander,?)
inputs/four_lander?????????	
/
fours&?#
inputs/fours?????????	
3
on_play(?%
inputs/on_play?????????	
-
ones%?"
inputs/ones?????????	
=
three_lander-?*
inputs/three_lander?????????	
1
threes'?$
inputs/threes?????????	
9

two_lander+?(
inputs/two_lander?????????	
-
twos%?"
inputs/twos?????????	
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_layer_call_fn_11236074????
???
???
(
cards?
cards?????????	
4
five_lander%?"
five_lander?????????	
4
four_lander%?"
four_lander?????????	
(
fours?
fours?????????	
,
on_play!?
on_play?????????	
&
ones?
ones?????????	
6
three_lander&?#
three_lander?????????	
*
threes ?
threes?????????	
2

two_lander$?!

two_lander?????????	
&
twos?
twos?????????	
p 

 
? "???????????
-__inference_sequential_layer_call_fn_11236291????
???
???
(
cards?
cards?????????	
4
five_lander%?"
five_lander?????????	
4
four_lander%?"
four_lander?????????	
(
fours?
fours?????????	
,
on_play!?
on_play?????????	
&
ones?
ones?????????	
6
three_lander&?#
three_lander?????????	
*
threes ?
threes?????????	
2

two_lander$?!

two_lander?????????	
&
twos?
twos?????????	
p

 
? "???????????
-__inference_sequential_layer_call_fn_11236597????
???
???
/
cards&?#
inputs/cards?????????	
;
five_lander,?)
inputs/five_lander?????????	
;
four_lander,?)
inputs/four_lander?????????	
/
fours&?#
inputs/fours?????????	
3
on_play(?%
inputs/on_play?????????	
-
ones%?"
inputs/ones?????????	
=
three_lander-?*
inputs/three_lander?????????	
1
threes'?$
inputs/threes?????????	
9

two_lander+?(
inputs/two_lander?????????	
-
twos%?"
inputs/twos?????????	
p 

 
? "???????????
-__inference_sequential_layer_call_fn_11236615????
???
???
/
cards&?#
inputs/cards?????????	
;
five_lander,?)
inputs/five_lander?????????	
;
four_lander,?)
inputs/four_lander?????????	
/
fours&?#
inputs/fours?????????	
3
on_play(?%
inputs/on_play?????????	
-
ones%?"
inputs/ones?????????	
=
three_lander-?*
inputs/three_lander?????????	
1
threes'?$
inputs/threes?????????	
9

two_lander+?(
inputs/two_lander?????????	
-
twos%?"
inputs/twos?????????	
p

 
? "???????????
&__inference_signature_wrapper_11236355????
? 
???
(
cards?
cards?????????	
4
five_lander%?"
five_lander?????????	
4
four_lander%?"
four_lander?????????	
(
fours?
fours?????????	
,
on_play!?
on_play?????????	
&
ones?
ones?????????	
6
three_lander&?#
three_lander?????????	
*
threes ?
threes?????????	
2

two_lander$?!

two_lander?????????	
&
twos?
twos?????????	"3?0
.
output_1"?
output_1?????????