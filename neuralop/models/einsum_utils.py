import torch


def einsum_complexhalf_two_input(eq, a, b):
    """
    Return the einsum(eq, a, b)
    We call this instead of standard einsum when either a or b is ComplexHalf,
    to run the operation in half precision.
    """
    assert len(eq.split(',')) == 2, "Einsum equation must have two inputs"

    # cast both tensors to real and half precision
    a = torch.view_as_real(a)
    b = torch.view_as_real(b)
    a = a.half()
    b = b.half()

    # create a new einsum equation 
    input_output = eq.split('->')
    new_output = 'xy' + input_output[1]
    input_terms = input_output[0].split(',')
    new_inputs = [input_terms[0] + 'x', input_terms[1] + 'y']
    new_eqn = new_inputs[0] + ',' + new_inputs[1] + '->' + new_output

    tmp = torch.einsum(new_eqn, a, b)
    res = torch.stack([tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1)
    return torch.view_as_complex(res)

def einsum_complexhalf(eq, *args):
    """Compute einsum for complexhalf tensors"""

    if len(args) == 2:
        return einsum_complexhalf_two_input(eq, *args)

    # todo: this can be made general. Call opt_einsum to get the partial_eqns
    assert eq == 'abcd,e,be,fe,ce,de->afcd', "Currently only implemented for this eqn"

    partial_eqns = ['fe,e->fe',
                    'de,be->deb',
                    'fe,ce->fec',
                    'fec,deb->fcdb',
                    'fcdb,abcd->afcd']

    tensors = {}
    labels = eq.split('->')[0].split(',')
    tensors = dict(zip(labels,args))

    for key, tensor in tensors.items():
        tensor = torch.view_as_real(tensor)
        tensor = tensor.half()
        tensors[key] = tensor

    # now all tensors are in the "view as real" form
    for partial_eq in partial_eqns:

        # get the tensors
        in_labels, out_label = partial_eq.split('->')
        in_labels = in_labels.split(',')
        in_tensors = [tensors[label] for label in in_labels]

        # create a new einsum equation 
        input_output = partial_eq.split('->')
        new_output = 'xy' + input_output[1]
        input_terms = input_output[0].split(',')
        new_inputs = [input_terms[0] + 'x', input_terms[1] + 'y']
        new_eqn = new_inputs[0] + ',' + new_inputs[1] + '->' + new_output

        # perform the einsum, and convert to "view as real" form
        tmp = torch.einsum(new_eqn, *in_tensors)
        result = torch.stack([tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1)
        tensors[out_label] = result

    return torch.view_as_complex(tensors['afcd'])