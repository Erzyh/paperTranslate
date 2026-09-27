; Inno Setup script for the paperTranslate installer.
; Built by desktop\package.ps1 (which passes AppVersion and the paths); do not
; compile it by hand without those defines.
;
; Per-user install into %LOCALAPPDATA%\Programs\paperTranslate: no admin
; rights, and the app folder stays writable so the built-in self-updater
; (backend\app\updater.py) can replace the files in place.

#ifndef AppVersion
  #error AppVersion must be defined (see desktop\package.ps1)
#endif

[Setup]
; Fixed id: keep it forever so upgrades and uninstall find the same install.
AppId={{8097B01D-4335-405A-9982-EAC7F0278B2C}
AppName=paperTranslate
AppVersion={#AppVersion}
AppVerName=paperTranslate {#AppVersion}
AppPublisher=Erzyh
AppPublisherURL=https://github.com/Erzyh/paperTranslate
AppSupportURL=https://github.com/Erzyh/paperTranslate/issues
AppUpdatesURL=https://github.com/Erzyh/paperTranslate/releases
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\paperTranslate
DisableProgramGroupPage=yes
DisableDirPage=auto
UninstallDisplayName=paperTranslate
UninstallDisplayIcon={app}\paperTranslate.exe
OutputDir={#OutputDir}
OutputBaseFilename=paperTranslate-Setup-v{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Close a running copy of the app before replacing its files.
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#AppDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\paperTranslate"; Filename: "{app}\paperTranslate.exe"
Name: "{autodesktop}\paperTranslate"; Filename: "{app}\paperTranslate.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\paperTranslate.exe"; Description: "{cm:LaunchProgram,paperTranslate}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Self-updates may add files the installer never recorded.
Type: filesandordirs; Name: "{app}"

[CustomMessages]
korean.RemoveUserData=번역 기록과 설정(API 키 포함)도 함께 지울까요?%n%n%1
english.RemoveUserData=Also delete translation history and settings (including API keys)?%n%n%1

[Code]
// Translations, keys and the job board live outside the app folder, in
// %LOCALAPPDATA%\paperTranslate. Offer to remove them on uninstall (never in
// a silent uninstall).
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\paperTranslate');
    if DirExists(DataDir) and not UninstallSilent() then
      if MsgBox(FmtMessage(CustomMessage('RemoveUserData'), [DataDir]),
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
