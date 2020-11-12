from django.db import models
from login_routes.models import Users, Ciudad


# Create your models here.
class Archivos(models.Model):

    description = models.TextField(verbose_name='Descripción de archivos', max_length=500)
    csv = models.FileField(verbose_name='Archivo CSV', upload_to='archivos/', null=True, blank=True)
    xlx = models.FileField(verbose_name='Archivo XLX', upload_to='archivos/', null=True, blank=True)

    class Meta:
        verbose_name = 'Archivo'

    def __str__(self):
        return str(self.id) + ' - Gestión de archivos para rutas'


"""------------------------------------------------------------------------------------------------------------"""


# Vehículos a usar en el sistema

class Vehiculo(models.Model):

    conductor = models.OneToOneField(
        Users,
        on_delete=models.SET_NULL,
        verbose_name='Conductor del vehículo',
        help_text='Indique el conductor asignado para este vehículo',
        null=True,
    )

    ciudad = models.ForeignKey(
        Ciudad,
        on_delete=models.SET_NULL,
        verbose_name='Ciudad donde se ubica',
        help_text='Indique en que ciudad se encuentra situado el vehículo a usar en el sistema.',
        null=True
    )

    direccion = models.CharField(
        max_length=200,
        verbose_name='Dirección de la ubicación del vehículo',
        help_text='Especifique la dirección donde el vehículo se encuentra ubicado',
        default='Calle 22 # 56-24'
    )

    capacidad = models.IntegerField(
        verbose_name='Capacidad de carga (en kg)',
        help_text='Estipule la maxima capacidad de carga en kilogramos para este vehiculo',
        default=5000
    )

    placa = models.CharField(
        verbose_name='Placa del vehículo',
        help_text='Ingrese la placa del vehículo a registrar para su identificación correcta',
        max_length=6,
        unique=True
    )

    REQUIRED_FIELDS = [
        'conductor',
        'capacidad',
        'placa'
    ]

    class Meta:
        verbose_name = 'Vehiculo'

    def __str__(self):
        return self.placa


# Clientes del sistema

class Cliente(models.Model):

    nombre = models.CharField(
        max_length=100,
        verbose_name='Nombre del cliente',
        help_text='Indique el nombre del cliente inscrito en este portal',
        blank=True
    )

    direccion = models.CharField(
        max_length=200,
        verbose_name='Dirección del cliente',
        help_text='Especifique la dirección donde el cliente se encuentra ubicado',
        blank=True
    )

    ciudad = models.ForeignKey(
        Ciudad,
        on_delete=models.CASCADE,
        verbose_name='Ciudad del cliente',
        help_text='Indique la ciudad del cliente'
    )

    REQUIRED_FIELDS = [
        'nombre',
        'direccion',
        'ciudad',
    ]

    class Meta:
        verbose_name = 'Cliente'

    def __str__(self):
        return self.nombre


# Pedido del cliente
class Pedido(models.Model):

    cliente = models.ForeignKey(
        Cliente,
        on_delete=models.CASCADE,
        verbose_name='Cliente dueño del pedido',
        help_text='Indique quien es el cliente propietario de este pedido'
    )

    cantidad_dummies = models.IntegerField(
        verbose_name='Cantidad de dummies',
        help_text='Indique la cantidad de dummies para este pedido',
        default=0,
        null=True
    )

    cantidad_giftcards = models.IntegerField(
        verbose_name='Cantidad de gift cards',
        help_text='Indique la cantidad de gift cards para este pedido',
        default=0,
        null=True
    )

    cantidad_handsets = models.IntegerField(
        verbose_name='Cantidad de handset',
        help_text='Indique la cantidad de handsets cards para este pedido',
        default=0,
        null=True
    )

    cantidad_modems = models.IntegerField(
        verbose_name='Cantidad de modems',
        help_text='Indique la cantidad de modems para este pedido',
        default=0,
        null=True
    )

    cantidad_packs = models.IntegerField(
        verbose_name='Cantidad de packs',
        help_text='Indique la cantidad de packs para este pedido',
        default=0,
        null=True
    )

    cantidad_prepaid = models.IntegerField(
        verbose_name='Cantidad de prepaid cards',
        help_text='Indique la cantidad de prepaid cards para este pedido',
        default=0,
        null=True
    )

    cantidad_simcards = models.IntegerField(
        verbose_name='Cantidad de sim cards',
        help_text='Indique la cantidad de sim cards para este pedido',
        default=0,
        null=True
    )

    cantidad_simcards_prep = models.IntegerField(
        verbose_name='Cantidad de sim cards prepagas',
        help_text='Indique la cantidad de sim cards prepagas para este pedido',
        default=0,
        null=True
    )

    cantidad_supplies = models.IntegerField(
        verbose_name='Cantidad de embalajes',
        help_text='Indique la cantidad de embalajes de productos para este pedido',
        default=0,
        null=True
    )

    REQUIRED_FIELDS = [
        'cliente',
    ]

    class Meta:

        verbose_name = 'Pedido'

    def __str__(self):
        return '[' + str(self.id) + '] ' + self.cliente.nombre


# Despacho del producto

class Despacho(models.Model):

    pedido = models.ForeignKey(
        Pedido,
        on_delete=models.SET_NULL,
        verbose_name='Pedido a despachar',
        help_text='Asigne el pedido que será despachado',
        null=True
    )

    vehiculo = models.ForeignKey(
        Vehiculo,
        on_delete=models.SET_NULL,
        verbose_name='Vehículo asignado',
        help_text='Indique cual será el vehículo destinado para realizar el despacho asignado',
        null=True
    )

    si_min = models.IntegerField(
        verbose_name='Tiempo de servicio',
        help_text='Es decir, vehículo llega a la dirección, se estaciona, entonces es el tiempo que se demora el '
                  'conductor o el ayudante en buscar el pedido en el vehículo y entregárselo al cliente',
        default=0
    )

    ai_min = models.IntegerField(
        verbose_name='Ventana de inicio',
        help_text='Es decir, el vehículo puede llegar  después del minuto x a entregar el pedido, en ese caso '
                  'particular fueron tomadas según la normativa y restricciones según las zonas de la ciudad para el '
                  'movimiento de los vehículos de carga, siendo las horas valle',
        default=0
    )

    bi_min = models.IntegerField(
        verbose_name='Ventana de fin',
        help_text='Es decir, es el tiempo máximo en el que puede llegar el vehículo para entregar el pedido',
        default=0
    )

    fecha_despacho = models.DateField(
        auto_now_add=True,
        verbose_name='Fecha de registro del despacho',
        help_text='Fecha autogenerada para el despacho el día que se registre'
    )

    REQUIRED_FIELDS = [
        'pedido',
    ]

    class Meta:
        verbose_name = 'Despacho'
        get_latest_by = 'id'

    def __str__(self):
        return str(self.id)
