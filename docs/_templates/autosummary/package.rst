{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :no-members:

{% if modules %}
Subpackages and modules
-----------------------

.. autosummary::
   :toctree:
   :recursive:

{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
