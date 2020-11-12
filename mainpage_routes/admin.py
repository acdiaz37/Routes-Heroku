from datetime import date
from django.contrib import admin

from .models import *
from . import routes

from import_export.admin import ImportExportModelAdmin, ExportMixin
from import_export.formats import base_formats
from import_export import resources, fields

from django.utils.html import format_html


# Parametrización del panel de cargue de archivos de ruta

class ArchivosAdmin(admin.ModelAdmin):

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    fieldsets = (
        (
            'Gestión de archivos para calculo de ruta', {
                'fields': ['description', 'csv', 'xlx', ]
            }
        ),
    )

    readonly_fields = ['description']


admin.site.register(Archivos, ArchivosAdmin)

"""------------------------------------------------------------------------------------------------------------"""


# Parametrización del panel administrativo para Ciudad

class CiudadAdmin(ImportExportModelAdmin):

    ordering = list_filter = list_display = ['nombre']

    fieldsets = (

        (
            'Registro de ciudad', {
                'fields': ['nombre', ]
            }
        ),

    )


# Parametrización del panel administrativo para Vehiculo

class VehiculoResource(resources.Resource):

    id_vehiculo = fields.Field(
        column_name='Vehicle',
        attribute='id',
    )
    cap_vehiculo = fields.Field(
        column_name='Capacity_[kg]',
        attribute='capacidad',
    )

    class Meta:

        model = Vehiculo

        exclude = [
            'conductor',
            'ciudad',
            'direccion',
            'placa',
        ]
        fields = [
            'id_vehiculo',
            'cap_vehiculo',
        ]
        export_order = fields


class VehiculoAdmin(ImportExportModelAdmin):

    resource_class = VehiculoResource

    def get_export_formats(self):
        formats = [base_formats.XLSX]
        return [f for f in formats if f().can_export()]

    list_display = ['id', 'placa', 'capacidad', 'get_full_address']
    list_filter = ['id', 'placa', 'ciudad__nombre']

    ordering = ['id']

    fieldsets = (

        (
            'Datos del vehículo', {
                'fields': ['conductor', 'placa']
            }
        ),
        (
            'Ubicación del vehículo', {
                'fields': ['ciudad', 'direccion']
            }
        ),
    )

    def get_full_address(self, obj):
        return str(obj.direccion) + ', ' + str(obj.ciudad)

    get_full_address.short_description = 'Ubicación del vehículo'


# Parametrización del panel administrativo para Cliente

class ClienteAdmin(ImportExportModelAdmin):

    list_display = ['id', 'nombre', 'direccion', 'ciudad']
    list_filter = ['ciudad__nombre', 'nombre']

    ordering = ['id']

    fieldsets = (

        (
            'Datos del cliente', {
                'fields': ['nombre']
            }
        ),
        (
            'Ubicación del cliente', {
                'fields': ['ciudad', 'direccion']
            }
        ),
    )


# Parametrización del panel administrativo para Pedido

class PedidoAdmin(ImportExportModelAdmin):

    list_display = [
        'id',
        'cliente',
        'get_direccion_pedido',
        'get_cant_items',
    ]

    list_filter = ['id', 'cliente__nombre', 'cliente__ciudad']

    ordering = ['id']

    fieldsets = (

        (
            'Cliente del pedido', {
                'fields': ['cliente']
            }
        ),
        (
            'Contenido del pedido', {
                'fields': [
                    'cantidad_dummies',
                    'cantidad_giftcards',
                    'cantidad_handsets',
                    'cantidad_modems',
                    'cantidad_packs',
                    'cantidad_prepaid',
                    'cantidad_simcards',
                    'cantidad_simcards_prep',
                    'cantidad_supplies'
                ]
            }
        ),
    )

    def get_direccion_pedido(self, obj):
        return str(obj.cliente.direccion) + ', ' + str(obj.cliente.ciudad)

    def get_cant_items(self, obj):

        total_items = obj.cantidad_dummies + obj.cantidad_giftcards + obj.cantidad_handsets + \
                      obj.cantidad_modems + obj.cantidad_packs + obj.cantidad_prepaid + \
                      obj.cantidad_simcards + obj.cantidad_simcards_prep + obj.cantidad_supplies

        return total_items

    get_direccion_pedido.short_description = 'Dirección del cliente'
    get_cant_items.short_description = 'Total de elementos'


# Parametrización del panel administrativo para Despacho

class DespachoResource(resources.ModelResource):

    id_cliente = fields.Field(column_name='ID')
    nombre_cliente = fields.Field(column_name='Client_Depot')
    latitude = fields.Field(column_name='Lat_[y]')
    longitude = fields.Field(column_name='Lon_[x]')
    total_weight = fields.Field(column_name='Q_[Kg]')
    si_min = fields.Field(column_name='Si_[min]', attribute='si_min')
    ai_min = fields.Field(column_name='ai_[min]', attribute='ai_min')
    bi_min = fields.Field(column_name='bi_[min]', attribute='bi_min')
    city = fields.Field(column_name='City')
    conductor = fields.Field(column_name='driver_id')

    class Meta:
        model = Despacho

        exclude = [
            'id',
            'pedido',
            'fecha_despacho',
        ]

        fields = [
            'id_cliente',
            'nombre_cliente',
            'latitude',
            'longitude',
            'total_weight',
            'si_min',
            'ai_min',
            'bi_min',
            'city',
            'conductor',
        ]

        export_order = fields

    @staticmethod
    def dehydrate_id_cliente(despacho):
        return despacho.pedido.cliente.id

    @staticmethod
    def dehydrate_nombre_cliente(despacho):
        return despacho.pedido.cliente.nombre

    @staticmethod
    def dehydrate_latitude(despacho):

        direccion = despacho.pedido.cliente.direccion
        ciudad = despacho.pedido.cliente.ciudad

        rts = routes.init()
        coord = rts.coords()

        my_coordinates = coord.get_coordinates(direccion, set_city=ciudad)

        return format(my_coordinates.get('lat'), '.6f')

    @staticmethod
    def dehydrate_longitude(despacho):

        direccion = despacho.pedido.cliente.direccion
        ciudad = despacho.pedido.cliente.ciudad

        rts = routes.init()
        coord = rts.coords()

        my_coordinates = coord.get_coordinates(direccion, set_city=ciudad)

        return format(my_coordinates.get('lng'), '.6f')

    @staticmethod
    def dehydrate_total_weight(despacho):

        get_weight = (despacho.pedido.cantidad_dummies * 0.0403) + (despacho.pedido.cantidad_giftcards * 0.0030) + \
                 (despacho.pedido.cantidad_handsets * 0.22) + (despacho.pedido.cantidad_modems * 0.04) + \
                 (despacho.pedido.cantidad_packs * 0.80) + (despacho.pedido.cantidad_prepaid * 0.0035) + \
                 (despacho.pedido.cantidad_simcards * 0.0035) + (despacho.pedido.cantidad_simcards_prep * 0.0035) + \
                 (despacho.pedido.cantidad_supplies * 0.03)

        return format(get_weight, '.0f')

    @staticmethod
    def dehydrate_city(despacho):
        return despacho.pedido.cliente.ciudad.nombre

    @staticmethod
    def dehydrate_conductor(despacho):
        return despacho.vehiculo.conductor_id


class DespachoAdmin(ExportMixin, admin.ModelAdmin):

    resource_class = DespachoResource

    def get_export_queryset(self, request):
        return Despacho.objects.filter(fecha_despacho=date.today())

    def get_export_formats(self):
        formats = [base_formats.CSV]
        return [f for f in formats if f().can_export()]

    list_display = [
        'get_pedido_id_cliente',
        'get_pedido_nombre_cliente',
        'get_latitude',
        'get_longitude',
        'get_total_weight',
        'si_min',
        'ai_min',
        'bi_min',
        'get_city',
        'fecha_despacho',
    ]

    list_filter = ['id', 'fecha_despacho', 'pedido__cliente__nombre']

    ordering = ['id', 'fecha_despacho', ]

    fieldsets = (
        (
            'Registro', {
                'fields': ['fecha_despacho', ]
            }
        ),
        (
            'Especificación del pedido', {
                'fields': ['pedido', ]
            }
        ),
        (
            'Asignación de responsable de despacho', {
                'fields': ['vehiculo', ]
            }
        ),
        (
            'Parametros para configuración de ruta del pedido', {
                'fields': ['si_min', 'ai_min', 'bi_min']
            }
        )

    )

    readonly_fields = ['fecha_despacho']

    def get_pedido_id_cliente(self, obj):
        return obj.pedido.cliente.id

    def get_pedido_nombre_cliente(self, obj):
        return obj.pedido.cliente.nombre

    def get_latitude(self, obj):

        direccion = obj.pedido.cliente.direccion
        ciudad = obj.pedido.cliente.ciudad

        rts = routes.init()
        coord = rts.coords()

        my_coordinates = coord.get_coordinates(direccion, set_city=ciudad)

        return format(my_coordinates.get('lat'), '.6f')

    def get_longitude(self, obj):

        direccion = obj.pedido.cliente.direccion
        ciudad = obj.pedido.cliente.ciudad

        rts = routes.init()
        coord = rts.coords()

        my_coordinates = coord.get_coordinates(direccion, set_city=ciudad)

        return format(my_coordinates.get('lng'), '.6f')

    def get_total_weight(self, obj):

        weight = (obj.pedido.cantidad_dummies * 0.0403) + (obj.pedido.cantidad_giftcards * 0.0030) + \
                 (obj.pedido.cantidad_handsets * 0.22) + (obj.pedido.cantidad_modems * 0.04) + \
                 (obj.pedido.cantidad_packs * 0.80) + (obj.pedido.cantidad_prepaid * 0.0035) + \
                 (obj.pedido.cantidad_simcards * 0.0035) + (obj.pedido.cantidad_simcards_prep * 0.0035) + \
                 (obj.pedido.cantidad_supplies * 0.03)

        return format(weight, '.0f')

    def get_city(self, obj):
        return obj.pedido.cliente.ciudad.nombre

    get_pedido_id_cliente.short_description = 'ID Cliente'
    get_pedido_nombre_cliente.short_description = 'Nombre cliente'
    get_latitude.short_description = 'Latitud'
    get_longitude.short_description = 'Longitud'
    get_total_weight.short_description = 'Peso total (kg)'
    get_city.short_description = 'Ciudad'


# Registro de parametrizaciones en el panel administrativo

admin.site.register(Ciudad, CiudadAdmin)
admin.site.register(Vehiculo, VehiculoAdmin)
admin.site.register(Cliente, ClienteAdmin)
admin.site.register(Pedido, PedidoAdmin)
admin.site.register(Despacho, DespachoAdmin)
