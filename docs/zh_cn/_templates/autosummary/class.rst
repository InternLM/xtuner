{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}
   {% endif %}

   .. autosummary::
   {% for item in methods %}
   {% if item != '__init__' %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}

   {% for item in methods %}
   {% if item != '__init__' %}
   .. automethod:: {{ item }}
   {% endif %}
   {% endfor %}

   {% endblock %}
