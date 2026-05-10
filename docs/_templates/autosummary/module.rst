{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

{% if modules %}
Submodules
----------

.. autosummary::
   :toctree:

{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
