from catan_ai import api


def test_cython_selects_local_checkpoint_only_when_present_and_unset(tmp_path):
    path = tmp_path / 'planner_parameter.cython.json'
    env = {}
    assert api.select_parameter_checkpoint(env, 'cython', path) is None and env == {}  # file absent
    path.write_text('{}')
    assert api.select_parameter_checkpoint(env, 'cython', path) == str(path)
    assert env['CATAN_PARAMETER_MODEL'] == str(path)
    explicit = {'CATAN_PARAMETER_MODEL': 'mine.json'}
    assert api.select_parameter_checkpoint(explicit, 'cython', path) is None  # explicit choice wins
    assert explicit['CATAN_PARAMETER_MODEL'] == 'mine.json'
    plain = {}
    assert api.select_parameter_checkpoint(plain, 'python', path) is None and plain == {}  # python keeps the committed one
