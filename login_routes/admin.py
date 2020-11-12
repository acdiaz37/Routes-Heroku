from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import Users as Conductor

from .forms import (
    UserAdminCreationForm,
    UserAdminChangeForm
)


# Register your models here.

class ConductorAdmin(UserAdmin):

    form = UserAdminChangeForm
    add_form = UserAdminCreationForm

    list_display = ['id', 'get_full_name', 'cedula', 'licencia']
    list_filter = ['id', 'cedula', 'licencia']

    add_fieldsets = [
        (
            'Credenciales de acceso', {
                'classes': ['wide', ],
                'fields': ['email', 'password1', 'password2', ]
            }
        ),
        (
            'Información personal del usuario', {
                'classes': ['wide', ],
                'fields': ['first_name', 'last_name', 'cedula', 'licencia', ]
            }
        ),
        (
            'Permisos en el sitio', {
                'fields': ['active', 'staff', 'admin', ]
            }
        )
    ]

    fieldsets = [
        (
            'Credenciales de acceso', {
                'classes': ['wide', ],
                'fields': ['email', 'password', ]
            }
        ),
        (
            'Información personal del usuario', {
                'classes': ['wide', ],
                'fields': ['first_name', 'last_name', 'cedula', 'licencia', ]
            }
        ),
        (
            'Permisos en el sitio', {
                'fields': ['active', 'staff', 'admin', ]
            }
        )
    ]

    def get_full_name(self, obj):
        return str(obj.first_name) + ' ' + str(obj.last_name)

    get_full_name.short_description = 'Nombre completo'

    ordering = ['id']
    filter_horizontal = []


admin.site.register(Conductor, ConductorAdmin)
